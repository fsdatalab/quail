"""Resolve columns and bind AI expressions in SQL queries."""

from sqlglot import exp

from quail.catalog import Catalog
from quail.frontend.label_tables import read_label_table
from quail.frontend.prompts import PromptBinder
from quail.frontend.scope import Scope
from quail.frontend.sql_binding.predicates import column_path
from quail.logical import (
    Alias,
    Compare,
    CompileError,
    ExtractStruct,
    FilterPredicate,
    JoinSpec,
    ModelCall,
    RegularPredicate,
)
from quail.logical.expressions import validate_task_description

JOIN_OPTION_KEYS = {"selectivity", "anchor"}


SCORE_OPTION_KEYS = {"selectivity"}


def _is_call(node, name: str) -> bool:
    return isinstance(node, exp.Anonymous) and str(node.this).upper() == name


_COMPARISON_CLASSES = {
    exp.LT: ("<", ">"),
    exp.LTE: ("<=", ">="),
    exp.GT: (">", "<"),
    exp.GTE: (">=", "<="),
}


def _score_comparison(node):
    """Return (comparison, AI_SCORE call, threshold) or None."""
    for kind, (left, right) in _COMPARISON_CLASSES.items():
        if not isinstance(node, kind):
            continue
        if _is_call(node.this, "AI_SCORE"):
            return left, node.this, node.expression
        if _is_call(node.expression, "AI_SCORE"):
            return right, node.expression, node.this
    return None


def _label_test(node):
    """Return the AI.CLASSIFY call and tested values of an = or IN test, or None."""
    if isinstance(node, exp.In) and isinstance(node.this, exp.AIClassify):
        return node.this, list(node.expressions)
    if isinstance(node, exp.EQ):
        left, right = node.this, node.expression
        if isinstance(left, exp.AIClassify):
            return left, [right]
        if isinstance(right, exp.AIClassify):
            return right, [left]
    return None


def _is_ai_score_comparison(node) -> bool:
    return _score_comparison(node) is not None


class ExpressionBinder:
    def __init__(self, catalog: Catalog, prompts: PromptBinder,
                 dialect: str = "snowflake"):
        self.catalog = catalog
        self.prompts = prompts
        self.dialect = dialect
        self.scope = Scope(catalog)
        self.tables = self.scope.tables
        self.regular_predicates: dict[str, list[RegularPredicate]] = {}
        self.document_columns = self.scope.document_columns
        self.filters: dict[str, list[FilterPredicate]] = {}
        self.label_tests: dict[
            str, list[tuple[ModelCall, tuple[str, ...], float | None]]
        ] = {}
        # alias -> (RegularPredicate on an answer or span, selectivity)
        self.extract_tests: dict[str, list[tuple[RegularPredicate, float | None]]] = {}
        self.joins: list[JoinSpec] = []

    # ---- scope ------------------------------------------------------

    def add_table(self, table: exp.Table) -> str:
        alias = table.alias or table.name
        self.scope.add_table(alias, table.name)
        return alias

    def resolve_value(self, node, tables=None, function=None):
        """Resolve a column, a struct field under one, or LENGTH() of either.

        Args:
            node: The column or LENGTH(column) expression.
            tables: (alias, provider) pairs to resolve against instead
                of the tables in scope.
            function: The function already parsed off the node, if any.

        Raises:
            CompileError: The node is not a column expression, or its
                column, field, or function does not resolve.
        """
        if isinstance(node, exp.Length) and function is None:
            node, function = node.this, "length"
        if not isinstance(node, exp.Column):
            raise CompileError(f"{node.sql()} is not a column expression")
        parts = column_path(node)
        scope = dict(self.tables if tables is None else tables)
        alias = parts[0] if len(parts) > 1 and parts[0] in scope else None
        column, *path = parts[1:] if alias is not None else parts
        return self.scope.resolve_value(column, alias, path=tuple(path),
                                        function=function, tables=tables,
                                        shown=node.sql())

    def note_document_column(self, column) -> None:
        self.scope.note_document_column(column)

    # ---- AI predicate parsing ----------------------------------------

    def parse_options(self, node, allowed: set) -> dict:
        if node is None:
            return {}
        if not isinstance(node, exp.Struct):
            raise CompileError(
                f"the second AI operator argument must be an option "
                f"object like {{'selectivity': 0.3}}, got {node.sql()}"
            )
        out = {}
        for prop in node.expressions:
            if not isinstance(prop, exp.PropertyEQ):
                raise CompileError(f"malformed option {prop.sql()}")
            key = str(prop.this.name)
            if key not in allowed:
                raise CompileError(
                    f"unknown option key {key!r}; allowed: {sorted(allowed)}"
                )
            value = prop.expression
            if key == "selectivity":
                if not (isinstance(value, exp.Literal) and not value.is_string):
                    raise CompileError("selectivity must be a number")
                out[key] = float(value.this)
            else:  # anchor
                if not (isinstance(value, exp.Literal) and value.is_string):
                    raise CompileError("anchor must be a table alias string")
                out[key] = str(value.this)
        return out

    def bind_ai_call(
        self, node, function: str, allowed: set, scope=None, join=None, classify=None
    ):
        """Parse one AI prompt call into prompt, options, and aliases.

        Args:
            node: The call.
            function: Its internal name.
            allowed: The option keys it takes.
            scope: The aliases in scope.
            join: Whether the prompt reads a pair; None decides from
                the aliases it reads.
            classify: For AI.CLASSIFY, the keyword arguments of
                bind_classify_prompt: labels, descriptions, and
                task_description.
        """
        if not _is_call(node, function):
            raise CompileError(
                f"only {function}(PROMPT(...)) predicates are "
                f"supported here, got: {node.sql()}"
            )
        args = node.expressions
        if not args or not _is_call(args[0], "PROMPT"):
            raise CompileError(
                f"{function}'s first argument must be PROMPT('template', columns...)"
            )
        if len(args) > 2:
            raise CompileError(f"{function} takes PROMPT and at most one option object")
        options = self.parse_options(args[1] if len(args) == 2 else None, allowed)
        p_args = args[0].expressions
        if not p_args or not (
            isinstance(p_args[0], exp.Literal) and p_args[0].is_string
        ):
            raise CompileError("PROMPT's first argument must be a string template")
        template = str(p_args[0].this)
        refs = []
        for a in p_args[1:]:
            if not isinstance(a, (exp.Column, exp.Length)):
                raise CompileError(
                    f"PROMPT arguments must be column references, got {a.sql()}"
                )
            refs.append(self.resolve_value(a, scope))
        aliases = []
        for r in refs:
            if r.alias not in aliases:
                aliases.append(r.alias)
        if join is None:
            join = len(aliases) > 1
        if function == "AI_SCORE" and join and template.count("{0}") != 1:
            raise CompileError(
                "an AI.SCORE pair prompt mentions {0} exactly once, as "
                "the place its document is inserted; refer to it again "
                "in words"
            )
        if classify is not None:
            prompt = self.prompts.bind("label", template, tuple(refs), **classify)
        else:
            if function == "AI_SCORE":
                kind = "score"
            else:
                kind = "join" if join else "filter"
            prompt = self.prompts.bind(kind, template, tuple(refs))
        for r in refs:
            self.note_document_column(r)
        return prompt, options, aliases

    def bind_ai_classify(self, node, scope=None):
        """Parse an AI.CLASSIFY expression and bind its document references.

        Args:
            node: SQLGlot AIClassify expression.
            scope: Optional set of table aliases allowed in this expression.

        Returns:
            A tuple of the label ModelCall, planner options, and referenced aliases.

        Raises:
            CompileError: The input, categories, options, or document references
                are invalid.
        """
        categories = node.args.get("categories")
        if isinstance(categories, exp.Kwarg):
            if str(categories.this.name).lower() != "categories":
                raise CompileError(
                    f"AI.CLASSIFY takes categories => ..., got {categories.this.name}"
                )
            categories = categories.expression
        labels, descriptions = self.parse_categories(categories)
        options = self.parse_classify_options(node.args.get("config"))
        prompt_call = node.this
        if isinstance(prompt_call, exp.Column):
            # a bare document column uses the default instruction
            prompt_call = exp.Anonymous(
                this="PROMPT", expressions=[exp.Literal.string("{0}"), prompt_call]
            )
        call = exp.Anonymous(this="AI_CLASSIFY", expressions=[prompt_call])
        prompt, _, aliases = self.bind_ai_call(
            call,
            "AI_CLASSIFY",
            set(),
            scope=scope,
            join=False,
            classify=dict(
                labels=labels,
                descriptions=descriptions,
                task_description=options.pop("task_description", ""),
            ),
        )
        if len(aliases) not in (1, 2):
            raise CompileError(
                "AI.CLASSIFY reads one document column, or one from each side of a join"
            )
        model_call = ModelCall(
            prompt,
            "label",
            labels,
            descriptions,
            probabilities=options.pop("probabilities", False),
        )
        model_call.validate()
        return model_call, options, aliases

    def bind_ai_extract(self, node, name: str) -> tuple:
        """Parse an AI.EXTRACT call and bind its document reference.

        The call is ``AI.EXTRACT(column, {'field': 'question', ...})``.
        Each field is its own extraction, named ``name.field``.

        Args:
            node: The AI_EXTRACT call.
            name: The struct column's AS name.

        Returns:
            The ExtractStruct and one extraction Alias per field, in
            written order.

        Raises:
            CompileError: The column or fields are invalid.
        """
        ref, fields = self._extract_arguments(node)
        questions = {}
        for prop in fields.expressions:
            if not isinstance(prop, exp.PropertyEQ):
                raise CompileError(f"malformed AI.EXTRACT field {prop.sql()}")
            field, question = str(prop.this.name), prop.expression
            if field in questions:
                raise CompileError(f"AI.EXTRACT {name!r} names {field!r} twice")
            if not (isinstance(question, exp.Literal) and question.is_string):
                raise CompileError(
                    f"AI.EXTRACT field {field!r} is a question string, "
                    f"got {question.sql()}")
            questions[field] = str(question.this)
        struct = ExtractStruct(name, tuple(questions))
        struct.validate()
        return struct, tuple(
            Alias(self._extract_call(ref, question), struct.member(field))
            for field, question in questions.items())

    def _extract_arguments(self, node) -> tuple:
        """Check an AI.EXTRACT call's arguments and resolve its column.

        Returns:
            The column reference and the object of field questions.
        """
        args = list(node.expressions)
        if len(args) != 2:
            raise CompileError(
                "AI.EXTRACT takes a document column and an object of field "
                "questions")
        if not isinstance(args[0], (exp.Column, exp.Length)):
            raise CompileError(
                f"AI.EXTRACT's first argument is a document column, "
                f"got {args[0].sql()}")
        if not isinstance(args[1], exp.Struct):
            raise CompileError(
                f"AI.EXTRACT's second argument is an object of field "
                f"questions such as {{'ends': 'When does the lease end?'}}, "
                f"got {args[1].sql()}")
        ref = self.resolve_value(args[0])
        return ref, args[1]

    def _extract_call(self, ref, question: str) -> ModelCall:
        """Bind one extraction's prompt over the document column."""
        prompt = self.prompts.bind("extract", question, (ref,))
        self.note_document_column(ref)
        call = ModelCall(prompt, "extract", question=question)
        call.validate()
        return call

    def bind_ai_value(self, node: exp.Expression) -> ModelCall:
        """Bind a score or classification to its model call."""
        if isinstance(node, exp.AIClassify):
            call, _, _ = self.bind_ai_classify(node)
            return call
        prompt, _, _ = self.bind_ai_call(node, "AI_SCORE", set())
        return ModelCall(prompt, "score")

    def parse_categories(self, node) -> tuple[tuple, tuple]:
        """Read the labels and descriptions of an AI.CLASSIFY categories argument.

        Args:
            node: Array of strings, label-description pairs, category objects,
                or an unqualified registered label table name.

        Returns:
            The labels and their descriptions, in the same order. Descriptions
            is empty when no label has one.

        Raises:
            CompileError: The categories have an invalid shape or value type,
                or the registered label table is invalid.
        """
        if isinstance(node, exp.Column) and not node.table:
            return read_label_table(self.catalog, str(node.name))
        if not isinstance(node, exp.Array):
            raise CompileError(
                "AI.CLASSIFY categories are an ARRAY of labels, of (label, "
                "description) pairs, or of {'label': ..., 'description': "
                "...} objects, or a registered label table's name"
            )
        labels = []
        descriptions = []
        for item in node.expressions:
            if isinstance(item, exp.Literal) and item.is_string:
                labels.append(str(item.this))
                descriptions.append("")
                continue
            if isinstance(item, exp.Tuple) and len(item.expressions) == 2:
                label, description = item.expressions
            elif isinstance(item, exp.Struct):
                fields = {}
                for prop in item.expressions:
                    if not isinstance(prop, exp.PropertyEQ):
                        raise CompileError(f"malformed category {item.sql()}")
                    fields[str(prop.this.name).lower()] = prop.expression
                unknown = set(fields) - {"label", "description"}
                if unknown or "label" not in fields:
                    raise CompileError(
                        f"a category object has a label and an optional "
                        f"description, got {item.sql()}"
                    )
                label = fields["label"]
                description = fields.get("description")
            else:
                raise CompileError(
                    f"AI.CLASSIFY category {item.sql()} is not a label, a "
                    f"(label, description) pair, or a category object"
                )
            if not (isinstance(label, exp.Literal) and label.is_string):
                raise CompileError("AI.CLASSIFY labels are string literals")
            if description is not None and not (
                isinstance(description, exp.Literal)
                and description.is_string
                or isinstance(description, exp.Null)
            ):
                raise CompileError(
                    "an AI.CLASSIFY description is a string literal or NULL"
                )
            labels.append(str(label.this))
            descriptions.append(
                ""
                if description is None or isinstance(description, exp.Null)
                else str(description.this)
            )
        if not any(descriptions):
            descriptions = []
        return tuple(labels), tuple(descriptions)

    def parse_classify_options(self, node) -> dict:
        """Validate and extract the options for AI.CLASSIFY.

        Args:
            node: SQLGlot Struct expression, or None for no options.

        Returns:
            A dictionary of supplied selectivity, task_description, and
            probabilities values. Defaults are applied by the caller.

        Raises:
            CompileError: The object is malformed, a key is unknown, a value has
                the wrong type, or the task description exceeds its word limit.
        """
        if node is None:
            return {}
        if not isinstance(node, exp.Struct):
            raise CompileError(
                f"AI.CLASSIFY's config is an object like "
                f"{{'task_description': '...'}}, got {node.sql()}"
            )
        out = {}
        for prop in node.expressions:
            if not isinstance(prop, exp.PropertyEQ):
                raise CompileError(f"malformed option {prop.sql()}")
            key = str(prop.this.name).lower()
            value = prop.expression
            text = (
                str(value.this)
                if isinstance(value, exp.Literal) and value.is_string
                else None
            )
            if key == "selectivity":
                if not (isinstance(value, exp.Literal) and not value.is_string):
                    raise CompileError("selectivity must be a number")
                out[key] = float(value.this)
            elif key == "task_description":
                if text is None:
                    raise CompileError("task_description is a string")
                validate_task_description(text)
                out[key] = text
            elif key == "probabilities":
                if not isinstance(value, exp.Boolean):
                    raise CompileError("probabilities is true or false")
                out[key] = bool(value.this)
            else:
                raise CompileError(
                    f"unknown AI.CLASSIFY option {key!r}; the options are "
                    f"selectivity, task_description, and "
                    f"probabilities"
                )
        return out

    def bind_ai_filter(self, node, allowed: set, scope=None, join=None):
        """Parse an AI_FILTER node into a boolean call, options, and aliases."""
        prompt, options, aliases = self.bind_ai_call(
            node, "AI_FILTER", allowed, scope=scope, join=join
        )
        return ModelCall(prompt, "boolean"), options, aliases

    def bind_ai_score(self, node, allowed: set, scope=None, join=None):
        """Parse a compared AI.SCORE call into a Compare, options, and aliases.

        The threshold may be on either side of the comparison.
        """
        parsed = _score_comparison(node)
        if parsed is None:
            raise CompileError("AI.SCORE must be compared with <, <=, >, or >=")
        comparison, call, threshold = parsed
        if not isinstance(threshold, exp.Literal) or threshold.is_string:
            raise CompileError("AI.SCORE threshold must be a number")
        value = float(threshold.this)
        if not 0.0 <= value <= 1.0:
            raise CompileError("AI.SCORE threshold must be between 0 and 1")
        prompt, options, aliases = self.bind_ai_call(
            call,
            "AI_SCORE",
            allowed,
            scope=scope,
            join=join,
        )
        expression = Compare(ModelCall(prompt, "score"), comparison, value)
        return expression, options, aliases


def _conjuncts(node) -> list:
    """Flatten one AND tree into its terms, in written order."""
    terms, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, exp.And):
            stack.append(n.expression)
            stack.append(n.this)
        else:
            terms.append(n)
    return terms  # the pop order above yields written order
