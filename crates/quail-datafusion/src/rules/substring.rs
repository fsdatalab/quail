use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr::Like;
use datafusion::logical_expr::expr_rewriter::unnormalize_col;
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, LogicalPlan};

use crate::ai::{AiNode, Semantics};

pub(crate) fn precheck(node: &AiNode, input: &LogicalPlan) -> Option<Expr> {
    if node.function() != "ai_extract" || node.semantics() != Semantics::Rows {
        return None;
    }
    let Expr::ScalarFunction(call) = node.call() else {
        return None;
    };
    let document = call.args.first()?;
    let filter = node.filter()?;
    let literal = split_conjunction(&filter)
        .into_iter()
        .filter_map(|term| required_literal(term, node))
        .max_by_key(|literal| literal.chars().count())?;
    let test = Expr::Like(Like::new(
        false,
        Box::new(document.clone()),
        Box::new(Expr::Literal(
            ScalarValue::Utf8(Some(format!("%{}%", escaped(&literal)))),
            None,
        )),
        Some('\\'),
        true,
    ));
    (!tests_below(input, &test)).then_some(test)
}

fn required_literal(term: &Expr, node: &AiNode) -> Option<String> {
    let Expr::Like(Like {
        negated: false,
        expr,
        pattern,
        escape_char,
        ..
    }) = term
    else {
        return None;
    };
    let Expr::ScalarFunction(field) = expr.as_ref() else {
        return None;
    };
    let reads_answer = field.name() == "get_field"
        && matches!(field.args.first(), Some(Expr::Column(column))
            if column.relation.is_none() && column.name == node.answer());
    if !reads_answer {
        return None;
    }
    let Expr::Literal(ScalarValue::Utf8(Some(pattern)), _) = pattern.as_ref() else {
        return None;
    };
    let run = literal_runs(pattern, escape_char.unwrap_or('\\'))
        .into_iter()
        .max_by_key(|run| run.chars().count())?;
    run.split_whitespace()
        .max_by_key(|word| word.chars().count())
        .map(ToString::to_string)
}

fn literal_runs(pattern: &str, escape: char) -> Vec<String> {
    let mut runs = vec![String::new()];
    let mut chars = pattern.chars();
    while let Some(found) = chars.next() {
        match found {
            '%' | '_' => runs.push(String::new()),
            found if found == escape => {
                if let Some(next) = chars.next() {
                    runs.last_mut().expect("a run").push(next);
                }
            }
            found => runs.last_mut().expect("a run").push(found),
        }
    }
    runs.retain(|run| !run.is_empty());
    runs
}

fn escaped(text: &str) -> String {
    let mut found = String::new();
    for character in text.chars() {
        if matches!(character, '%' | '_' | '\\') {
            found.push('\\');
        }
        found.push(character);
    }
    found
}

fn tests_below(plan: &LogicalPlan, test: &Expr) -> bool {
    let named = |expr: &Expr| unnormalize_col(expr.clone());
    let wanted = named(test);
    let mut found = false;
    let _ = plan.apply(|node| {
        let predicates: Vec<&Expr> = match node {
            LogicalPlan::Filter(filter) => split_conjunction(&filter.predicate),
            LogicalPlan::TableScan(scan) => scan.filters.iter().collect(),
            LogicalPlan::Projection(_) | LogicalPlan::SubqueryAlias(_) => Vec::new(),
            _ => return Ok(TreeNodeRecursion::Stop),
        };
        if predicates
            .into_iter()
            .any(|predicate| named(predicate) == wanted)
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patterns_literal_runs_skip_its_wildcards_and_escapes() {
        assert_eq!(literal_runs("%20_7%", '\\'), ["20", "7"]);
        assert_eq!(literal_runs("a\\%b%c", '\\'), ["a%b", "c"]);
        assert_eq!(literal_runs("%%", '\\'), Vec::<String>::new());
        assert_eq!(escaped("50%_\\"), "50\\%\\_\\\\");
    }
}
