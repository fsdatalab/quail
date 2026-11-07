//! Build the physical graph of one plan's operators in the chosen operator
//! order, as `quail/planner/graph/builder.py` and the operators'
//! `emit_*` functions build it.

use std::collections::{BTreeMap, BTreeSet};

use quail_cost::budgets::pages;
use quail_cost::quantity::python_sum;
use quail_cost::work::Work;
use quail_physical::{
    AiClassifyAttributes, AiExtractAttributes, AiFilterAttributes, AiJoinAttributes,
    AiMapAttributes, AiSaeFeaturesAttributes, AiScoreAttributes, Asked, BarrierAttributes,
    ClassifySpec, ColumnRef, ExchangeAttributes, Expression, FilterAttributes, HashJoinAttributes,
    ModelCall, Node, PhysicalNode, PortRef, Prompt, ScanAttributes, ScoreSpec, ValueExpression,
    input_ports, port,
};
use serde_json::{Map, Value};

use crate::classify::classify_statistics;
use crate::columns::{ColumnSpec, column_spec};
use crate::context::PlanningContext;
use crate::costs::question_tokens;
use crate::expressions::asked_call;
use crate::expressions::{
    CompileError, Refusal, SPAN_SUFFIX, argument_alias, call_aliases, condition_columns, repr_str,
};
use crate::operators::{ColumnTest, Operators, SemanticColumn};
use crate::pricing::PlanPricer;
use crate::results::limit_node;
use crate::retention::{group_sequence, streamed_aliases};
use crate::scores::score_spec;
use crate::search::{
    ASK, OperatorOrder, StepKey, TableStep, choose_operator_order, column_step_kind, label_calls,
    resident_after,
};
use crate::statistics::{arena_capacity, effective_selectivity, pair_counts};
use crate::validation::{
    columns_above_joins, has_score, is_score, score_projections, validate_query,
};

/// A physical plan: its graph, settings, and estimates, as `PhysicalPlan`.
#[derive(Debug, Clone)]
pub struct PhysicalPlan {
    pub model: String,
    pub device: String,
    pub gpus: i64,
    pub backend: String,
    pub estimated_seconds: f64,
    pub nodes: Vec<PhysicalNode>,
    pub settings: Map<String, Value>,
    pub root: PortRef,
    /// node id -> its explain entry: its seconds, and what releasing a
    /// filter sequence's KV would cost.
    pub estimates: Map<String, Value>,
    /// What prices an edited graph again.
    pub pricer: PlanPricer,
}

impl PhysicalPlan {
    /// The plan as it crosses to the executor.
    pub fn envelope(&self) -> quail_physical::PlanEnvelope {
        quail_physical::PlanEnvelope {
            backend: self.backend.clone(),
            model: self.model.clone(),
            device: self.device.clone(),
            gpus: self.gpus,
            settings: self.settings.clone(),
            graph: quail_physical::PhysicalGraph {
                nodes: self.nodes.clone(),
                root: self.root.clone(),
            },
        }
    }
}

/// A plan, or why the configuration cannot run it.
pub type Planned = Result<PhysicalPlan, Refusal>;

fn round_one(value: f64) -> f64 {
    // Python's round(value, 1): the decimal nearest the exact value
    format!("{value:.1}").parse().expect("a float")
}

fn tagged_node<A>(id: &str, inputs: &[PortRef], attributes: A) -> Node<A> {
    Node {
        id: id.to_string(),
        inputs: input_ports(inputs),
        attributes,
    }
}

/// One HashJoin per join with equality conditions, over the scans, as
/// `hash_join_nodes`.
pub fn hash_join_nodes(
    found: &Operators,
    pair_fractions: &BTreeMap<usize, f64>,
    scan_ports: &BTreeMap<String, PortRef>,
) -> Vec<PhysicalNode> {
    let mut nodes = Vec::new();
    for (position, join) in found.joins.iter().enumerate() {
        let conditions = &join.on;
        if conditions.is_empty() {
            continue;
        }
        let left = argument_alias(&conditions[0].left);
        let right = argument_alias(&conditions[0].right);
        let on: Vec<(ValueExpression, ValueExpression)> = conditions
            .iter()
            .map(|condition| {
                if argument_alias(&condition.left) != left {
                    (condition.right.clone(), condition.left.clone())
                } else {
                    (condition.left.clone(), condition.right.clone())
                }
            })
            .collect();
        nodes.push(PhysicalNode::HashJoin(tagged_node(
            &format!("hash_join:{left}-{right}"),
            &[scan_ports[&left].clone(), scan_ports[&right].clone()],
            HashJoinAttributes {
                left: left.clone(),
                right: right.clone(),
                on,
                written_position: position as i64,
                pair_fraction: pair_fractions.get(&position).copied().unwrap_or(1.0),
            },
        )));
    }
    nodes
}

/// Split ordered documents into compact token balanced ranges, as
/// `contiguous_shards`.
pub fn contiguous_shards(
    lengths: &[i64],
    gpus: i64,
    total: Option<i64>,
) -> (Vec<(i64, i64)>, Vec<i64>) {
    let num_documents = lengths.len() as i64;
    if let (1, Some(total)) = (gpus, total) {
        return (vec![(0, num_documents)], vec![total]);
    }
    let mut cumulative = vec![0i64];
    for length in lengths {
        cumulative.push(cumulative.last().expect("a sum") + length);
    }
    let total = *cumulative.last().expect("a sum");
    let zeros: Vec<i64> = lengths
        .iter()
        .enumerate()
        .filter(|(_, length)| **length == 0)
        .map(|(index, _)| index as i64)
        .collect();
    // numpy's searchsorted on the left: the first index whose value is at least x
    let left = |values: &[i64], x: f64| {
        values
            .iter()
            .position(|v| *v as f64 >= x)
            .unwrap_or(values.len()) as i64
    };
    let right =
        |values: &[i64], x: i64| values.iter().position(|v| *v > x).unwrap_or(values.len()) as i64;
    let (mut ranges, mut loads) = (Vec::new(), Vec::new());
    let (mut start, mut consumed) = (0i64, 0i64);
    for worker in 0..gpus {
        let (stop, load);
        if worker == gpus - 1 {
            stop = num_documents;
            load = total - consumed;
        } else {
            let remaining = (gpus - worker) as f64;
            let target = (total - consumed) as f64 / remaining;
            let mut found =
                num_documents.min((start + 1).max(left(&cumulative, consumed as f64 + target)));
            let before = cumulative[start.max(found - 1) as usize] - consumed;
            let after = cumulative[found as usize] - consumed;
            if before > 0 && (target - before as f64).abs() <= (target - after as f64).abs() {
                found -= 1;
            }
            // the greedy rule stops before a zero-length document once load is positive
            let positive = right(&cumulative, consumed);
            let zero = left(&zeros, positive as f64);
            if (zero as usize) < zeros.len() {
                found = found.min(zeros[zero as usize]);
            }
            stop = found;
            load = cumulative[stop as usize] - consumed;
        }
        ranges.push((start, stop));
        loads.push(load);
        start = stop;
        consumed += load;
    }
    (ranges, loads)
}

/// The longest input of a score: its resident prefix and streamed suffix
/// tokens, as `score_input_tokens`.
fn score_input_tokens(
    spec: &ScoreSpec,
    context: &PlanningContext,
) -> Result<(i64, i64), CompileError> {
    let parts = &spec.prompt_token_parts;
    let canvas = context.model.canvas_tokens;
    let mut longest = Vec::new();
    for alias in &spec.aliases {
        longest.push(context.lengths(alias)?.iter().copied().max().unwrap_or(0));
    }
    if spec.aliases.len() == 1 {
        return Ok((
            parts[0].len() as i64,
            longest[0] + parts[1].len() as i64 + canvas,
        ));
    }
    let prefix = parts[0].len() as i64 + longest[0] + parts[1].len() as i64;
    Ok((prefix, longest[1] + parts[2].len() as i64 + canvas))
}

/// What the AI joins leave for the operators after them, as `JoinPorts`.
#[derive(Debug, Default)]
struct JoinPorts {
    node_ids: BTreeMap<usize, String>,
    answers: Vec<PortRef>,
    aliases: Vec<String>,
    score_pairs: Option<PortRef>,
}

/// The score operators of a plan, as `ScoreLowering`.
struct ScoreLowering {
    projected: Vec<(String, ModelCall)>,
    chunk: i64,
    capacity: i64,
    names: Vec<(Prompt, String)>,
    work: Work,
    index: usize,
}

impl ScoreLowering {
    fn name_of(&self, prompt: &Prompt) -> Option<&String> {
        self.names
            .iter()
            .find(|(found, _)| found == prompt)
            .map(|(_, name)| name)
    }
}

/// The step kinds of one table's steps.
type Kinds = BTreeSet<&'static str>;

struct GraphBuilder<'a, 'c> {
    context: &'a PlanningContext<'c>,
    order: OperatorOrder,
    operators: &'a Operators,
    classified: BTreeSet<String>,
    after_joins: BTreeMap<String, Kinds>,
    classified_after: BTreeSet<String>,
    ask_filters: BTreeSet<String>,
    stop_keys: BTreeMap<String, Vec<String>>,
    sequence_groups: Vec<Vec<usize>>,
    group_ids: Vec<String>,
    seen_ids: BTreeMap<String, usize>,
    streamed: BTreeSet<String>,
    pipelined: BTreeSet<String>,
    nodes: Vec<PhysicalNode>,
    computed_ports: Vec<((String, String), PortRef)>,
    computed_columns: BTreeMap<String, BTreeSet<String>>,
    column_nodes: Vec<((String, String), SemanticColumn)>,
    emitted_tests: Vec<ColumnTest>,
    scoring: Option<ScoreLowering>,
    pairs_src: BTreeMap<usize, String>,
    label_names: Vec<(ModelCall, String)>,
}

impl<'a, 'c> GraphBuilder<'a, 'c> {
    fn new(
        found: &'a Operators,
        context: &'a PlanningContext<'c>,
        order: OperatorOrder,
    ) -> Result<Self, CompileError> {
        let labels = label_calls(found);
        let classified = labels.iter().map(|(_, alias)| alias.clone()).collect();
        let mut after_joins: BTreeMap<String, Kinds> = BTreeMap::new();
        for (alias, kind) in columns_above_joins(found) {
            after_joins
                .entry(alias)
                .or_default()
                .insert(column_step_kind(&kind));
        }
        let classified_after = after_joins
            .iter()
            .filter(|(_, kinds)| kinds.contains("classify"))
            .map(|(alias, _)| alias.clone())
            .collect();
        let ask_filters = order
            .statistics
            .asks
            .iter()
            .filter(|(_, positions)| !positions.is_empty())
            .map(|(alias, _)| alias.clone())
            .collect();
        let label_names = found
            .classifies
            .iter()
            .map(|column| (column.call.clone(), column.name.clone()))
            .collect();
        Ok(GraphBuilder {
            context,
            stop_keys: found.stop_keys.clone(),
            order,
            operators: found,
            classified,
            after_joins,
            classified_after,
            ask_filters,
            sequence_groups: Vec::new(),
            group_ids: Vec::new(),
            seen_ids: BTreeMap::new(),
            streamed: BTreeSet::new(),
            pipelined: BTreeSet::new(),
            nodes: Vec::new(),
            computed_ports: Vec::new(),
            computed_columns: BTreeMap::new(),
            column_nodes: Vec::new(),
            emitted_tests: Vec::new(),
            scoring: None,
            pairs_src: BTreeMap::new(),
            label_names,
        })
    }

    fn build(mut self) -> Result<Planned, CompileError> {
        self.prepare()?;
        let ports = self.emit_scans();
        let emitted = (|| -> Result<_, CompileError> {
            let ports = self.emit_documents(ports)?;
            let (mut ports, joins) = self.emit_joins(ports)?;
            self.emit_join_answers(&joins)?;
            self.emit_after_joins(&mut ports)?;
            let score_pairs = self.emit_pair_scores(&ports, &joins)?;
            Ok((ports, joins, score_pairs))
        })();
        let (ports, joins, score_pairs) = match emitted {
            Ok(found) => found,
            Err(CompileError(_, Some(refusal))) => return Ok(Err(*refusal)),
            Err(error) => return Err(error),
        };
        self.emit_result(&ports, &joins, score_pairs);
        self.finish().map(Ok)
    }

    fn prepare(&mut self) -> Result<(), CompileError> {
        let sequence = &self.order.join_sequence;
        self.sequence_groups = group_sequence(sequence);
        for group in &self.sequence_groups {
            let anchor = sequence[group[0]].1.clone();
            let seen = self.seen_ids.entry(anchor.clone()).or_insert(0);
            *seen += 1;
            self.group_ids.push(if *seen == 1 {
                format!("ai_join:{anchor}")
            } else {
                format!("ai_join:{anchor}:{seen}")
            });
        }
        self.streamed = streamed_aliases(
            sequence,
            &self.sequence_groups,
            &self.ask_filters,
            self.context.gpus,
        );
        for column in self.operators.columns() {
            if call_aliases(&column.call).len() != 1 {
                continue;
            }
            let alias = column.alias();
            self.computed_columns
                .entry(alias.clone())
                .or_default()
                .extend(column.added_columns());
            let key = (alias, column.name.clone());
            match self
                .column_nodes
                .iter_mut()
                .find(|(found, _)| *found == key)
            {
                Some(slot) => slot.1 = column.clone(),
                None => self.column_nodes.push((key, column.clone())),
            }
        }
        if has_score(self.operators) {
            let chunk = self.order.statistics.chunk;
            self.scoring = Some(ScoreLowering {
                projected: score_projections(self.operators),
                chunk,
                capacity: arena_capacity(self.context, chunk)?,
                names: Vec::new(),
                work: Work::default(),
                index: 0,
            });
        }
        Ok(())
    }

    fn emit_scans(&mut self) -> BTreeMap<String, PortRef> {
        let stats = &self.order.statistics.stats;
        let mut ports = BTreeMap::new();
        for scan in &self.operators.scans {
            let lengths = &self.context.document_tokens[&scan.alias];
            let total: i64 = lengths.iter().sum();
            let (shard_ranges, loads) = contiguous_shards(lengths, self.context.gpus, Some(total));
            let scan_id = format!("scan:{}", scan.alias);
            self.nodes.push(PhysicalNode::Scan(tagged_node(
                &scan_id,
                &[],
                ScanAttributes {
                    alias: scan.alias.clone(),
                    input_id: scan.alias.clone(),
                    num_documents: lengths.len() as i64,
                    total_tokens: total,
                    shard_ranges,
                    shard_token_loads: loads,
                    predicates: scan.predicates.clone(),
                    expected_rows: (!scan.predicates.is_empty())
                        .then(|| stats[&scan.alias].num_documents as f64),
                },
            )));
            ports.insert(
                scan.alias.clone(),
                port(&scan_id, format!("rows:{}", scan.alias)),
            );
        }
        for node in hash_join_nodes(self.operators, &self.context.pair_fractions, &ports) {
            if let PhysicalNode::HashJoin(join) = &node {
                self.pairs_src
                    .insert(join.attributes.written_position as usize, join.id.clone());
            }
            self.nodes.push(node);
        }
        // aliases whose filter sequence runs in one pipeline with the
        // classification or join after it
        let between: BTreeSet<&String> =
            self.classified.difference(&self.classified_after).collect();
        self.pipelined = self.streamed.clone();
        if self.context.gpus == 1 {
            for alias in &self.ask_filters {
                if between.contains(alias) {
                    self.pipelined.insert(alias.clone());
                }
            }
        }
        ports
    }

    fn emit_documents(
        &mut self,
        ports: BTreeMap<String, PortRef>,
    ) -> Result<BTreeMap<String, PortRef>, CompileError> {
        let mut ports = ports;
        let aliases: Vec<String> = self
            .operators
            .scans
            .iter()
            .map(|scan| scan.alias.clone())
            .collect();
        for alias in aliases {
            // a column operator the planner lifted above the joins is emitted after them
            let lifted = self.after_joins.get(&alias).cloned().unwrap_or_default();
            let kinds: Kinds = [ASK, "classify", "extract", "features", "map"]
                .into_iter()
                .filter(|kind| !lifted.contains(kind))
                .collect();
            let rows = self.order.statistics.stats[&alias].num_documents as f64;
            let after = self.emit_steps(&alias, rows, &kinds, ports[&alias].clone())?;
            ports.insert(alias, after);
        }
        Ok(ports)
    }

    /// Emit the given kinds of a table's steps in run order.
    fn emit_steps(
        &mut self,
        alias: &str,
        expected_rows: f64,
        kinds: &Kinds,
        port: PortRef,
    ) -> Result<PortRef, CompileError> {
        let steps: Vec<TableStep> = self
            .order
            .step_orders
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, steps)| {
                steps
                    .iter()
                    .filter(|step| kinds.contains(step.kind))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let mut runs: Vec<Vec<usize>> = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            match runs.last_mut() {
                Some(run) if steps[run[0]].kind == step.kind => run.push(index),
                _ => runs.push(vec![index]),
            }
        }
        let mut port = port;
        let mut expected_rows = expected_rows;
        // the scores SELECT returns follow the score predicates, before
        // the classifications and extractions read the rows
        let mut scores_pending = kinds.contains(ASK) && self.scoring.is_some();
        let mut previous: Option<usize> = None;
        for run in &runs {
            let kind = steps[run[0]].kind;
            let resident =
                previous.is_some_and(|previous| resident_after(&steps[previous], &steps[run[0]]));
            if kind != ASK && scores_pending {
                port = self.emit_projected_scores(alias, expected_rows, port)?;
                scores_pending = false;
            }
            if kind == ASK {
                let positions: Vec<usize> = run
                    .iter()
                    .map(|index| match &steps[*index].key {
                        StepKey::Position(position) => *position,
                        StepKey::Name(_) => unreachable!("an ask has a position"),
                    })
                    .collect();
                port = if self.scoring.is_some() {
                    self.emit_scores(alias, &positions, expected_rows, port)?
                } else {
                    let followed = *run.last().expect("a step") != steps.len() - 1;
                    self.emit_filter(alias, &positions, expected_rows, port, followed)?
                };
            } else {
                let columns: Vec<SemanticColumn> = run
                    .iter()
                    .map(|index| match &steps[*index].key {
                        StepKey::Name(name) => self
                            .column_nodes
                            .iter()
                            .find(|((owner, found), _)| owner == alias && found == name)
                            .map(|(_, node)| node.clone())
                            .expect("a column operator"),
                        StepKey::Position(_) => unreachable!("a column step has a name"),
                    })
                    .collect();
                port = self.emit_columns(alias, expected_rows, port, &columns, resident)?;
            }
            for index in run {
                expected_rows *= steps[*index].selectivity;
            }
            previous = run.last().copied();
        }
        if scores_pending {
            port = self.emit_projected_scores(alias, expected_rows, port)?;
        }
        Ok(port)
    }

    fn filter_order(&self, alias: &str) -> &[usize] {
        self.order
            .filter_orders
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, positions)| positions.as_slice())
            .unwrap_or(&[])
    }

    fn filters_of(&self, alias: &str) -> &[quail_physical::FilterPredicate] {
        self.operators
            .filters
            .iter()
            .find(|(found, _)| found == alias)
            .map(|(_, predicates)| predicates.as_slice())
            .unwrap_or(&[])
    }

    /// Append a run of the alias's AI filters in their chosen order, as
    /// `emit_filter`.
    fn emit_filter(
        &mut self,
        alias: &str,
        positions: &[usize],
        expected_rows: f64,
        port: PortRef,
        followed: bool,
    ) -> Result<PortRef, CompileError> {
        let order = self.filter_order(alias).to_vec();
        let mut surviving = expected_rows;
        let mut port = port;
        for position in positions {
            let predicate = self.filters_of(alias)[*position].clone();
            let ask_id = if Some(position) == order.first() {
                format!("ai_filter:{alias}")
            } else {
                format!("ai_filter:{alias}:{position}")
            };
            let lowered = self
                .context
                .lowered(&asked_call(&predicate.expression).prompt)?;
            let model = &self.context.model;
            self.nodes.push(PhysicalNode::AiFilter(tagged_node(
                &ask_id,
                std::slice::from_ref(&port),
                AiFilterAttributes {
                    alias: alias.to_string(),
                    arena_writes: model.reuses_kv()
                        && (order.len() > 1 || followed || self.pipelined.contains(alias)),
                    keep_kv: false,
                    written_position: *position as i64,
                    question_tokens: question_tokens(&lowered, model.canvas_tokens)?,
                    preamble_tokens: lowered.preamble_token_ids.len() as i64,
                    selectivity: predicate.selectivity,
                    expected_rows: round_one(surviving),
                    question_token_ids: Vec::new(),
                    document_head: lowered.preamble_token_ids.clone(),
                    share_prefixes: false,
                    attention: String::new(),
                    stop_key: self.stop_keys.get(alias).cloned().unwrap_or_default(),
                    pass_group: -1,
                    state_tail_tokens: 0,
                    question_header_token_ids: Vec::new(),
                    document_ends: Value::Array(Vec::new()),
                },
            )));
            surviving *= effective_selectivity(predicate.selectivity);
            port = port_of(&ask_id, format!("rows:{alias}"));
        }
        Ok(port)
    }

    /// Append one score node, as `ScoreLowering.score`.
    #[allow(clippy::too_many_arguments)]
    fn score(
        &mut self,
        prompt: &Prompt,
        fallback: &str,
        inputs: &[PortRef],
        expected_rows: f64,
        mean_tokens: f64,
        pair_fraction: f64,
        prefix_groups: Option<f64>,
    ) -> Result<PortRef, CompileError> {
        let context = self.context;
        let scoring = self.scoring.as_ref().expect("a score plan");
        let name = scoring
            .projected
            .iter()
            .find(|(_, call)| *call.prompt == *prompt)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| fallback.to_string());
        let (spec, work) = score_spec(
            prompt,
            &name,
            expected_rows,
            Some(mean_tokens),
            context,
            scoring.chunk,
            pair_fraction,
            prefix_groups,
        )?;
        let (prefix, suffix) = score_input_tokens(&spec, context)?;
        let what = if spec.aliases.len() == 1 {
            format!("a document in {}", repr_str(&spec.aliases[0]))
        } else {
            format!(
                "one pair of {} and {}",
                repr_str(&spec.aliases[0]),
                repr_str(&spec.aliases[1])
            )
        };
        let (mut needed, mut available, mut unit) = (prefix + suffix, scoring.chunk, "tokens");
        // a set-attention pass keeps no KV, so only the chunk bounds it
        if needed <= available && context.model.reuses_kv() {
            let page_tokens = context.device.page_tokens;
            needed = pages(prefix, page_tokens) + pages(suffix, page_tokens);
            available = scoring.capacity.div_euclid(page_tokens);
            unit = "pages";
        }
        if needed > available {
            return Err(CompileError::refused(Refusal::new(
                format!(
                    "{what} needs {needed} {unit} with its prompt, but the execution budget is {available} {unit}"
                ),
                "suffix_over_chunk",
                needed as f64,
                available as f64,
                unit,
            )));
        }
        let scoring = self.scoring.as_mut().expect("a score plan");
        let node_id = format!("ai-score:{}", scoring.index);
        scoring.index += 1;
        let rows_port = if spec.aliases.len() == 1 {
            format!("rows:{}", spec.aliases[0])
        } else {
            "rows".to_string()
        };
        self.nodes.push(PhysicalNode::AiScore(tagged_node(
            &node_id,
            inputs,
            AiScoreAttributes {
                backend_name: "quail".into(),
                model: context.model.name.clone(),
                spec: Some(spec),
                keep_kv: false,
                pass_group: -1,
                document_ends: Value::Array(Vec::new()),
            },
        )));
        match scoring.names.iter_mut().find(|(found, _)| found == prompt) {
            Some(slot) => slot.1 = name,
            None => scoring.names.push((prompt.clone(), name)),
        }
        scoring.work = &scoring.work + &work;
        Ok(port_of(&node_id, rows_port))
    }

    /// Append the Filter comparing a score with a constant, as
    /// `ScoreLowering.comparison`.
    fn comparison(
        &mut self,
        asked: &Asked,
        selectivity: Option<f64>,
        source: PortRef,
        aliases: &[String],
        position: usize,
    ) -> PortRef {
        let alias = if aliases.len() == 1 {
            aliases[0].as_str()
        } else {
            "join"
        };
        let Asked::Compare(test) = asked else {
            unreachable!("a score is compared with a constant")
        };
        let call = asked_call(asked);
        let scoring = self.scoring.as_ref().expect("a score plan");
        let name = scoring
            .name_of(&call.prompt)
            .expect("a scored prompt")
            .clone();
        let node_id = format!("filter:{alias}:{position}");
        let rows_port = if aliases.len() == 1 {
            format!("rows:{}", aliases[0])
        } else {
            "rows".to_string()
        };
        self.nodes.push(PhysicalNode::Filter(tagged_node(
            &node_id,
            std::slice::from_ref(&source),
            FilterAttributes {
                predicate: Some(match &test.test {
                    ValueExpression::Value(compared) => ValueExpression::Value(Expression {
                        columns: vec![ColumnRef {
                            alias: alias.to_string(),
                            provider: String::new(),
                            column: name,
                        }],
                        ..compared.clone()
                    }),
                    other => other.clone(),
                }),
                aliases: aliases.to_vec(),
                selectivity,
                written_position: position as i64,
            },
        )));
        port_of(&node_id, rows_port)
    }

    fn scored(&self, prompt: &Prompt) -> bool {
        self.scoring
            .as_ref()
            .is_some_and(|scoring| scoring.name_of(prompt).is_some())
    }

    fn mean_tokens(&self, alias: &str) -> f64 {
        self.order.statistics.stats[alias].mean_document_tokens()
    }

    /// Append a run of the alias's score predicates, as `emit_scores`.
    fn emit_scores(
        &mut self,
        alias: &str,
        positions: &[usize],
        expected_rows: f64,
        port: PortRef,
    ) -> Result<PortRef, CompileError> {
        let mut port = port;
        let mut expected_rows = expected_rows;
        for position in positions {
            let predicate = self.filters_of(alias)[*position].clone();
            let prompt = (*asked_call(&predicate.expression).prompt).clone();
            let mut source = port;
            if !self.scored(&prompt) {
                source = self.score(
                    &prompt,
                    &format!("__score_{alias}_{position}"),
                    &[source],
                    expected_rows,
                    self.mean_tokens(alias),
                    1.0,
                    None,
                )?;
            }
            port = self.comparison(
                &predicate.expression,
                predicate.selectivity,
                source,
                &[alias.to_string()],
                *position,
            );
            expected_rows *= effective_selectivity(predicate.selectivity);
        }
        Ok(port)
    }

    /// Append the alias's scores returned in SELECT, as `emit_projected_scores`.
    fn emit_projected_scores(
        &mut self,
        alias: &str,
        expected_rows: f64,
        port: PortRef,
    ) -> Result<PortRef, CompileError> {
        let projected = self
            .scoring
            .as_ref()
            .expect("a score plan")
            .projected
            .clone();
        let mut port = port;
        for (name, call) in projected {
            if call_aliases(&call) != [alias.to_string()] || self.scored(&call.prompt) {
                continue;
            }
            port = self.score(
                &call.prompt,
                &name,
                &[port],
                expected_rows,
                self.mean_tokens(alias),
                1.0,
                None,
            )?;
        }
        Ok(port)
    }

    /// Append the score of one pair prompt, as `emit_score_pair`.
    fn emit_score_pair(
        &mut self,
        prompt: &Prompt,
        position: usize,
        pair_fraction: f64,
        pair_inputs: &[PortRef],
        ports: &BTreeMap<String, PortRef>,
    ) -> Result<PortRef, CompileError> {
        let mut aliases: Vec<String> = Vec::new();
        for argument in &prompt.args {
            let alias = argument_alias(argument);
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
        let (expected, prefixes) = pair_counts(&self.order.expected_rows, &aliases, pair_fraction);
        let mut inputs: Vec<PortRef> = aliases.iter().map(|alias| ports[alias].clone()).collect();
        inputs.extend(pair_inputs.iter().cloned());
        let mean = python_sum(aliases.iter().map(|alias| self.mean_tokens(alias)));
        self.score(
            prompt,
            &format!("__score_join_{position}"),
            &inputs,
            expected,
            mean,
            pair_fraction,
            Some(prefixes),
        )
    }

    fn pair_inputs(&self, position: usize) -> Vec<PortRef> {
        self.pairs_src
            .get(&position)
            .map(|producer| vec![port_of(producer, format!("pairs:{position}"))])
            .unwrap_or_default()
    }

    /// Append the scores and comparisons of one group of score joins.
    fn emit_score_joins(
        &mut self,
        group: &[usize],
        ports: &BTreeMap<String, PortRef>,
        score_pairs: Option<PortRef>,
    ) -> Result<Option<PortRef>, CompileError> {
        let mut score_pairs = score_pairs;
        for member in group {
            let spec = self.order.join_sequence[*member].0.clone();
            let position = spec.written_position;
            let join = self.operators.joins[position].clone();
            let pair_inputs = self.pair_inputs(position);
            let source = match score_pairs {
                Some(found) => found,
                None => self.emit_score_pair(
                    &asked_call(&join.predicate).prompt,
                    position,
                    spec.pair_fraction,
                    &pair_inputs,
                    ports,
                )?,
            };
            score_pairs = Some(self.comparison(
                &join.predicate,
                join.selectivity,
                source,
                &spec.aliases,
                position,
            ));
        }
        Ok(score_pairs)
    }

    /// Emit one table's column operators in run order, each followed by its
    /// tests, as `GraphBuilder.emit_columns`.
    fn emit_columns(
        &mut self,
        alias: &str,
        expected_rows: f64,
        port: PortRef,
        columns: &[SemanticColumn],
        resident: bool,
    ) -> Result<PortRef, CompileError> {
        let computed = self
            .computed_columns
            .get(alias)
            .cloned()
            .unwrap_or_default();
        let mut port = port;
        let mut expected_rows = expected_rows;
        for (index, column) in columns.iter().enumerate() {
            let node =
                self.column_operator(column, port.clone(), expected_rows, resident || index > 0)?;
            let rows_port = node.rows_port();
            let node_id = node.id().to_string();
            let mut names = vec![];
            match &node {
                PhysicalNode::AiExtract(extract) => {
                    let spec = extract.attributes.spec.as_ref().expect("a spec");
                    names.push(spec.name.clone());
                    names.push(format!("{}{SPAN_SUFFIX}", spec.name));
                }
                PhysicalNode::AiClassify(classify) => {
                    names.push(
                        classify
                            .attributes
                            .spec
                            .as_ref()
                            .expect("a spec")
                            .name
                            .clone(),
                    );
                }
                PhysicalNode::AiMap(map) => {
                    names.push(map.attributes.spec.as_ref().expect("a spec").name.clone());
                }
                PhysicalNode::AiSaeFeatures(features) => {
                    names.push(
                        features
                            .attributes
                            .spec
                            .as_ref()
                            .expect("a spec")
                            .name
                            .clone(),
                    );
                }
                _ => unreachable!("a column operator"),
            }
            self.nodes.push(node);
            port = port_of(&node_id, rows_port);
            for name in &names {
                self.set_computed(alias, name, port.clone());
            }
            let done: BTreeSet<String> = self
                .computed_ports
                .iter()
                .filter(|((owner, _), _)| owner == alias)
                .map(|((_, name), _)| name.clone())
                .collect();
            let tests: Vec<ColumnTest> = self
                .operators
                .column_filters
                .iter()
                .find(|(found, _)| found == alias)
                .map(|(_, tests)| tests.clone())
                .unwrap_or_default();
            for test in tests {
                let read: BTreeSet<String> = condition_columns(&test.condition)
                    .into_iter()
                    .filter(|name| computed.contains(name))
                    .collect();
                if self.emitted_tests.contains(&test) || !read.is_subset(&done) {
                    continue;
                }
                self.emitted_tests.push(test.clone());
                let filter_id = format!("filter:{alias}:{}", test.position);
                let inputs = self.filter_inputs(alias, &test.condition, &port);
                self.nodes.push(PhysicalNode::Filter(Node {
                    id: filter_id.clone(),
                    inputs: input_ports(&inputs),
                    attributes: FilterAttributes {
                        predicate: Some(test.condition.clone()),
                        aliases: vec![alias.to_string()],
                        selectivity: test.selectivity,
                        written_position: test.position as i64,
                    },
                }));
                port = port_of(&filter_id, format!("rows:{alias}"));
                expected_rows *= effective_selectivity(test.selectivity);
            }
            // the port after the tests still carries the operator's columns
            for name in &names {
                self.set_computed(alias, name, port.clone());
            }
        }
        Ok(port)
    }

    fn set_computed(&mut self, alias: &str, name: &str, port: PortRef) {
        let key = (alias.to_string(), name.to_string());
        match self
            .computed_ports
            .iter_mut()
            .find(|(found, _)| *found == key)
        {
            Some(slot) => slot.1 = port,
            None => self.computed_ports.push((key, port)),
        }
    }

    /// A document Filter's input ports: its rows, then each other column's.
    fn filter_inputs(
        &self,
        alias: &str,
        condition: &ValueExpression,
        port: &PortRef,
    ) -> Vec<PortRef> {
        let mut inputs = vec![port.clone()];
        for name in condition_columns(condition) {
            let found = self
                .computed_ports
                .iter()
                .find(|((owner, column), _)| owner == alias && *column == name)
                .map(|(_, found)| found);
            if let Some(found) = found
                && found != port
                && !inputs[1..].contains(found)
            {
                inputs.push(found.clone());
            }
        }
        inputs
    }

    fn count_of(&self, type_name: &str) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.type_name() == type_name)
            .count()
    }

    /// One column operator's physical node, as its ColumnPlanner's `node`.
    fn column_operator(
        &mut self,
        column: &SemanticColumn,
        port: PortRef,
        expected_rows: f64,
        resident: bool,
    ) -> Result<PhysicalNode, CompileError> {
        let context = self.context;
        let model = context.model.name.clone();
        let inputs = std::slice::from_ref(&port);
        if column.call.kind == "label" {
            let table = classify_statistics(context, &column.alias(), &[])?;
            let (spec, work) = table.prepare(
                &column.call,
                &column.name,
                expected_rows,
                context.tokenizer,
                context.decode,
            )?;
            if let Some(scoring) = self.scoring.as_mut() {
                scoring.work = &scoring.work + &work;
            }
            let index = self.count_of("quail.ai_classify");
            return Ok(classify_node(
                &format!("ai-classify:{index}"),
                inputs,
                spec,
                model,
            ));
        }
        let spec = column_spec(context, column, expected_rows, resident)?;
        Ok(match spec {
            ColumnSpec::Extract(spec) => {
                let index = self.count_of("quail.ai_extract");
                PhysicalNode::AiExtract(tagged_node(
                    &format!("ai-extract:{index}"),
                    inputs,
                    AiExtractAttributes {
                        backend_name: "quail".into(),
                        model,
                        spec: Some(spec),
                        keep_kv: false,
                        pass_group: -1,
                        document_ends: Value::Array(Vec::new()),
                        attention: String::new(),
                    },
                ))
            }
            ColumnSpec::Map(spec) => {
                let index = self.count_of("quail.ai_map");
                PhysicalNode::AiMap(tagged_node(
                    &format!("ai-map:{index}"),
                    inputs,
                    AiMapAttributes {
                        backend_name: "quail".into(),
                        model,
                        spec: Some(spec),
                        keep_kv: false,
                        pass_group: -1,
                        document_ends: Value::Array(Vec::new()),
                        attention: String::new(),
                    },
                ))
            }
            ColumnSpec::SaeFeatures(spec) => {
                let index = self.count_of("quail.ai_sae_features");
                PhysicalNode::AiSaeFeatures(tagged_node(
                    &format!("ai-sae-features:{index}"),
                    inputs,
                    AiSaeFeaturesAttributes {
                        backend_name: "quail".into(),
                        model,
                        spec: Some(spec),
                        keep_kv: false,
                        pass_group: -1,
                        document_ends: Value::Array(Vec::new()),
                    },
                ))
            }
        })
    }

    fn unique_id(&mut self, prefix: &str, name: &str) -> String {
        let key = format!("{prefix}:{name}");
        let seen = self.seen_ids.entry(key.clone()).or_insert(0);
        *seen += 1;
        if *seen == 1 {
            key
        } else {
            format!("{key}:{seen}")
        }
    }

    /// The tokens before each anchor document, as `join_document_head`.
    fn join_document_head(&self, prompt: &Prompt, anchor: &str) -> Result<Vec<i64>, CompileError> {
        let lowered = self.context.lowered(prompt)?;
        let mut head = lowered.preamble_token_ids.clone();
        if self.context.model.attention_mask == "set" {
            let label = lowered
                .label_token_ids
                .iter()
                .find(|(alias, _, _)| alias == anchor)
                .map(|(_, label, _)| label.clone())
                .expect("a label for the anchor");
            head.extend(label);
        }
        Ok(head)
    }

    /// Create the joins and the exchanges between anchor groups.
    fn emit_joins(
        &mut self,
        ports: BTreeMap<String, PortRef>,
    ) -> Result<(BTreeMap<String, PortRef>, JoinPorts), CompileError> {
        let mut ports = ports;
        let mut joins = JoinPorts::default();
        let mut exec_index = 0;
        let groups = self.sequence_groups.clone();
        for (g, group) in groups.iter().enumerate() {
            let anchor = self.order.join_sequence[group[0]].1.clone();
            if self.order.join_sequence[group[0]].0.score.is_some() {
                joins.score_pairs =
                    self.emit_score_joins(group, &ports, joins.score_pairs.take())?;
                continue;
            }
            if g > 0 {
                let ahead: Vec<String> = self
                    .operators
                    .scans
                    .iter()
                    .map(|scan| scan.alias.clone())
                    .collect();
                let barrier_id = self.unique_id("barrier", &anchor);
                let mut inputs = joins.answers.clone();
                inputs.extend(ahead.iter().map(|alias| ports[alias].clone()));
                self.nodes.push(PhysicalNode::Barrier(tagged_node(
                    &barrier_id,
                    &inputs,
                    BarrierAttributes {
                        next_anchor: anchor.clone(),
                        aliases: ahead.clone(),
                    },
                )));
                for alias in &ahead {
                    ports.insert(alias.clone(), port_of(&barrier_id, format!("rows:{alias}")));
                }
            }
            if self.context.gpus > 1 {
                // anchors go to the GPU holding their KV, else balanced
                let exchange_id = self.unique_id("exchange", &anchor);
                self.nodes.push(PhysicalNode::Exchange(tagged_node(
                    &exchange_id,
                    &[ports[&anchor].clone()],
                    ExchangeAttributes {
                        anchor: anchor.clone(),
                    },
                )));
                ports.insert(
                    anchor.clone(),
                    port_of(&exchange_id, format!("rows:{anchor}")),
                );
            }
            let group_id = self.group_ids[g].clone();
            for (rank, member) in group.iter().enumerate() {
                let spec = self.order.join_sequence[*member].0.clone();
                let step = self.order.join_steps[*member].clone();
                let position = spec.written_position;
                let join_id = if rank == 0 {
                    group_id.clone()
                } else {
                    format!("{group_id}:predicate:{position}")
                };
                joins.node_ids.insert(position, join_id.clone());
                let partners: Vec<String> = spec
                    .aliases
                    .iter()
                    .filter(|a| **a != anchor)
                    .cloned()
                    .collect();
                let mut inputs: Vec<PortRef> = std::iter::once(&anchor)
                    .chain(&partners)
                    .map(|alias| ports[alias].clone())
                    .collect();
                let pair_inputs = self.pair_inputs(position);
                let pairs_from = self.pairs_src.get(&position).cloned().unwrap_or_default();
                inputs.extend(pair_inputs);
                let prompt = asked_call(&self.operators.joins[position].predicate)
                    .prompt
                    .clone();
                self.nodes.push(PhysicalNode::AiJoin(tagged_node(
                    &join_id,
                    &inputs,
                    AiJoinAttributes {
                        anchor: anchor.clone(),
                        anchor_resident: step.resident.to_string(),
                        keep_kv: false,
                        written_position: position as i64,
                        exec_index,
                        partners,
                        semantics: spec.semantics.clone(),
                        selectivity: spec.selectivity,
                        expected_rows: round_one(step.tuples),
                        anchor_documents: round_one(step.anchor_documents),
                        anchor_frame_tokens: spec.frame_tokens[&anchor],
                        pair_tail_tokens: spec.tail_tokens,
                        tuple_tokens: round_one(step.tokens),
                        pairs_from,
                        frame_token_ids: Vec::new(),
                        label_token_ids: Vec::new(),
                        tail_token_ids: Vec::new(),
                        document_head: self.join_document_head(&prompt, &anchor)?,
                        attention: String::new(),
                        share_prefixes: false,
                        separator_tokens: 0,
                        question_header_token_ids: Vec::new(),
                        document_ends: Value::Array(Vec::new()),
                    },
                )));
                ports.insert(anchor.clone(), port_of(&join_id, format!("rows:{anchor}")));
                if spec.semantics == "full" {
                    joins
                        .answers
                        .push(port_of(&join_id, format!("join_answers:{position}")));
                    for alias in &spec.aliases {
                        if !joins.aliases.contains(alias) {
                            joins.aliases.push(alias.clone());
                        }
                    }
                }
                exec_index += 1;
            }
        }
        Ok((ports, joins))
    }

    /// Append the classifications of joined rows after their joins, as
    /// `emit_joined_classify`.
    fn emit_join_answers(&mut self, joins: &JoinPorts) -> Result<(), CompileError> {
        let context = self.context;
        for (call, _) in label_calls(self.operators) {
            let aliases = call_aliases(&call);
            if aliases.len() != 2 {
                continue;
            }
            let wanted: BTreeSet<&String> = aliases.iter().collect();
            let mut stage = None;
            for group in &self.sequence_groups {
                let anchor = &self.order.join_sequence[group[0]].1;
                if let Some(member) = group.iter().find(|member| {
                    let spec = &self.order.join_sequence[**member].0;
                    spec.aliases.iter().collect::<BTreeSet<_>>() == wanted
                }) {
                    stage = Some((anchor.clone(), self.order.join_sequence[*member].0.clone()));
                    break;
                }
            }
            let (anchor, stage) = stage.expect("a classification of joined rows without its join");
            let partner = aliases
                .iter()
                .find(|alias| **alias != anchor)
                .expect("a partner")
                .clone();
            let table = classify_statistics(context, &anchor, &[])?;
            let position = stage.written_position;
            let join = &self.operators.joins[position];
            let expected = &self.order.expected_rows;
            let pairs = effective_selectivity(join.selectivity)
                * context
                    .pair_fractions
                    .get(&position)
                    .copied()
                    .unwrap_or(1.0)
                * expected[&anchor]
                * expected[&partner];
            let name = self
                .label_names
                .iter()
                .find(|(found, _)| *found == call)
                .map(|(_, name)| name.clone())
                .expect("a named classification");
            let (spec, _) = table.classify_joined(
                &call,
                &name,
                &partner,
                pairs,
                self.mean_tokens(&partner),
                Some(expected[&anchor]),
                context.tokenizer,
            )?;
            let index = self.count_of("quail.ai_classify");
            let node = classify_node(
                &format!("ai-classify:{index}"),
                &[port_of(
                    &joins.node_ids[&position],
                    format!("join_answers:{position}"),
                )],
                spec,
                context.model.name.clone(),
            );
            self.nodes.push(node);
        }
        Ok(())
    }

    /// Create the column operators the planner lifted above the joins.
    fn emit_after_joins(
        &mut self,
        ports: &mut BTreeMap<String, PortRef>,
    ) -> Result<(), CompileError> {
        // a column operator after the joins runs over the documents the
        // joins matched: a document with at least one true pair, expected as the
        // join selectivity times the partners it met, at most every document
        for (alias, kinds) in self.after_joins.clone() {
            let expected = &self.order.expected_rows;
            let met = python_sum(self.operators.joins.iter().enumerate().filter_map(
                |(position, join)| {
                    let arguments = &asked_call(&join.predicate).prompt.args;
                    if !arguments
                        .iter()
                        .any(|argument| argument_alias(argument) == alias)
                    {
                        return None;
                    }
                    let partners = python_sum(
                        arguments
                            .iter()
                            .filter(|argument| argument_alias(argument) != alias)
                            .map(|argument| expected[&argument_alias(argument)]),
                    );
                    Some(
                        effective_selectivity(join.selectivity)
                            * self
                                .context
                                .pair_fractions
                                .get(&position)
                                .copied()
                                .unwrap_or(1.0)
                            * partners,
                    )
                },
            ));
            let rows = expected[&alias] * met.min(1.0);
            let after = self.emit_steps(&alias, rows, &kinds, ports[&alias].clone())?;
            ports.insert(alias, after);
        }
        Ok(())
    }

    /// Create the pair scores SELECT returns.
    fn emit_pair_scores(
        &mut self,
        ports: &BTreeMap<String, PortRef>,
        joins: &JoinPorts,
    ) -> Result<Option<PortRef>, CompileError> {
        let mut score_pairs = joins.score_pairs.clone();
        let projected = match &self.scoring {
            Some(scoring) => scoring.projected.clone(),
            None => Vec::new(),
        };
        for (_, call) in projected {
            if call_aliases(&call).len() == 2 && !self.scored(&call.prompt) {
                score_pairs = Some(self.emit_score_pair(&call.prompt, 0, 1.0, &[], ports)?);
            }
        }
        Ok(score_pairs)
    }

    /// End the plan with the Limit of the region's fetch, which reads
    /// the region's rows: the scored pairs, the one full join's answers,
    /// or the surviving rows of the first table.
    fn emit_result(
        &mut self,
        ports: &BTreeMap<String, PortRef>,
        joins: &JoinPorts,
        score_pairs: Option<PortRef>,
    ) {
        let rows = match (score_pairs, joins.answers.first()) {
            (Some(pairs), _) => pairs,
            (None, Some(answers)) => answers.clone(),
            (None, None) => ports[&self.operators.scans[0].alias].clone(),
        };
        self.nodes.extend(limit_node(self.operators, &rows));
    }

    /// Build the plan and price it.
    fn finish(self) -> Result<PhysicalPlan, CompileError> {
        let context = self.context;
        let statistics = &self.order.statistics;
        // a score join's work is in its AiScore's estimate
        let stage_works: BTreeMap<usize, Work> = self
            .order
            .join_steps
            .iter()
            .filter(|step| !is_score(&self.operators.joins[step.written_position].predicate))
            .map(|step| (step.written_position, step.copy_work(context.gpus)))
            .collect();
        let pricer = PlanPricer {
            chunk: statistics.chunk,
            cap_pages: statistics.cap_pages,
            stats: statistics.stats.clone(),
            stage_works,
            expected_rows: self.order.expected_rows.clone(),
        };
        let model = &context.model;
        let mut settings = Map::new();
        settings.insert("chunk_tokens".into(), Value::from(statistics.chunk));
        settings.insert(
            "arena_pages".into(),
            Value::Array(vec![
                Value::from(statistics.arena_pages.0),
                Value::from(statistics.arena_pages.1),
            ]),
        );
        settings.insert("admission_tokens".into(), Value::from(statistics.admission));
        settings.insert(
            "order_rule".into(),
            Value::from(self.order.order_rule.clone()),
        );
        settings.insert(
            "order_source".into(),
            Value::from(self.order.order_source.clone()),
        );
        if let Some(scoring) = &self.scoring {
            settings.insert("retained_kv_tokens".into(), Value::from(scoring.capacity));
            settings.insert(
                "prefix_reuse".into(),
                Value::from("fixed prompt and first document within each score"),
            );
            settings.insert(
                "survivor_assumption".into(),
                Value::from("uniform independent selection"),
            );
            settings.insert(
                "estimated_fresh_tokens".into(),
                Value::from(scoring.work.tokens.total()),
            );
            settings.insert(
                "estimated_attention_pairs".into(),
                Value::from(scoring.work.pairs.total()),
            );
            settings.insert("batching".into(), Value::from("token_based_admission"));
            settings.insert("data_parallel_copies".into(), Value::from(context.gpus));
            settings.insert(
                "score_normalization".into(),
                Value::from(match model.role.as_str() {
                    "reranker" => "yes_no_softmax",
                    "decision" => "decision_head_softmax",
                    _ => "true_false_softmax",
                }),
            );
        }
        if !label_calls(self.operators).is_empty() {
            settings.insert(
                "classify_placement".into(),
                Value::from(if self.classified_after.is_empty() {
                    "before joins"
                } else {
                    "after joins"
                }),
            );
            if model.answer_canvas.is_some() {
                settings.insert("canvas_draws".into(), Value::from(context.canvas_draws));
            }
        }
        let last = self.nodes.last().expect("a node");
        let root = port_of(last.id(), last.outputs()[0].0.clone());
        let (seconds, estimates) = pricer.price(context, &self.nodes)?;
        Ok(PhysicalPlan {
            model: model.name.clone(),
            device: context.device.name.clone(),
            gpus: context.gpus,
            backend: "quail".into(),
            estimated_seconds: seconds,
            nodes: self.nodes,
            settings,
            root,
            estimates,
            pricer,
        })
    }
}

fn port_of(node_id: &str, name: impl Into<String>) -> PortRef {
    port(node_id, name)
}

fn classify_node(id: &str, inputs: &[PortRef], spec: ClassifySpec, model: String) -> PhysicalNode {
    PhysicalNode::AiClassify(tagged_node(
        id,
        inputs,
        AiClassifyAttributes {
            backend_name: "quail".into(),
            model,
            spec: Some(spec),
            keep_kv: false,
            pass_group: -1,
            document_ends: Value::Array(Vec::new()),
        },
    ))
}

/// Compile a plan's operators into a physical plan or a refusal, as
/// `build_physical_plan`.
///
/// # Errors
///
/// A callback fails, or the plan is malformed.
pub fn build_physical_plan(
    found: &Operators,
    context: &PlanningContext,
) -> Result<Planned, CompileError> {
    if let Some(refusal) = validate_query(found, context)? {
        return Ok(Err(refusal));
    }
    let order = match choose_operator_order(found, context)? {
        Ok(order) => order,
        Err(refusal) => return Ok(Err(refusal)),
    };
    GraphBuilder::new(found, context, order)?.build()
}

/// A refusal when a pair classification cannot run in its join's
/// pipeline, as `joined_classification_refusal`.
pub fn joined_classification_refusal(nodes: &[PhysicalNode]) -> Option<Refusal> {
    let pipelines = quail_physical::build_pipelines(nodes, &[]);
    for node in nodes {
        let PhysicalNode::AiClassify(classify) = node else {
            continue;
        };
        let Some(spec) = &classify.attributes.spec else {
            continue;
        };
        if spec.aliases.len() != 2 {
            continue;
        }
        let in_pipeline = pipelines.get(node.id()).is_some_and(|members| {
            members.iter().any(|member| {
                matches!(nodes[*member], PhysicalNode::AiJoin(_))
                    && node
                        .inputs()
                        .iter()
                        .any(|input| input.source.node_id == nodes[*member].id())
            })
        });
        if in_pipeline {
            continue;
        }
        return Some(Refusal::new(
            format!(
                "AI.CLASSIFY {} over a document pair must run in its join's pipeline; this plan \
                 requires unsupported standalone pair classification",
                repr_str(&spec.name)
            ),
            "joined_classify_pipeline",
            1.0,
            0.0,
            "supported join pipelines",
        ));
    }
    None
}

/// Put tokenized prompts and answer tokens in the physical plan, as
/// `bind_runtime_data`.
///
/// # Errors
///
/// A filter prompt has no token ids, the layout cannot run a
/// set-attention question, or the tokenizer fails.
pub fn bind_runtime_data(
    plan: &mut PhysicalPlan,
    found: &Operators,
    context: &PlanningContext,
) -> Result<(), CompileError> {
    let model = &context.model;
    let set_attention = model.attention_mask == "set";
    let layout = crate::prompts::prompt_layout(&model.prompt_layout)?;
    for node in &mut plan.nodes {
        match node {
            PhysicalNode::AiFilter(filter) => {
                let attributes = &mut filter.attributes;
                let predicates = &found
                    .filters
                    .iter()
                    .find(|(alias, _)| *alias == attributes.alias)
                    .expect("the alias's filters")
                    .1;
                let lowered = context.lowered(
                    &asked_call(&predicates[attributes.written_position as usize].expression)
                        .prompt,
                )?;
                if lowered.tail_token_ids.is_empty() {
                    return Err(CompileError::new("filter prompts have no token ids"));
                }
                attributes.question_token_ids = lowered.tail_token_ids.clone();
                if set_attention {
                    if !layout.label_segment {
                        return Err(CompileError::new(format!(
                            "{}: a set-attention layout tokenizes its question label as its own segment",
                            repr_str(&model.name)
                        )));
                    }
                    attributes.state_tail_tokens =
                        (context.tokenizer)(&lowered.prompt.tail_segments[0])?.len() as i64;
                    attributes.question_header_token_ids =
                        (context.tokenizer)(layout.question_label)?;
                }
            }
            PhysicalNode::AiJoin(join) => {
                let attributes = &mut join.attributes;
                let prompt = context.lowered(
                    &asked_call(&found.joins[attributes.written_position as usize].predicate)
                        .prompt,
                )?;
                let ids = |alias: &str| {
                    prompt
                        .label_token_ids
                        .iter()
                        .rev()
                        .find(|(found, _, _)| found == alias)
                        .map(|(_, label, frame)| (label.clone(), frame.clone()))
                        .expect("a label per alias")
                };
                attributes.frame_token_ids = ids(&attributes.anchor).1;
                attributes.label_token_ids = attributes
                    .partners
                    .iter()
                    .map(|partner| (partner.clone(), ids(partner).0))
                    .collect();
                attributes.tail_token_ids = prompt.tail_token_ids.clone();
                if set_attention {
                    if layout.record_label.is_none() {
                        return Err(CompileError::new(format!(
                            "{}: a set-attention layout labels each join record with its table",
                            repr_str(&model.name)
                        )));
                    }
                    attributes.separator_tokens =
                        (context.tokenizer)(&prompt.prompt.tail_segments[0])?.len() as i64;
                    attributes.question_header_token_ids =
                        (context.tokenizer)(layout.question_label)?;
                }
            }
            _ => {}
        }
    }
    let (true_ids, false_ids) =
        crate::lowering::answer_ids(&context.prompt_model(), context.tokenizer)?;
    plan.settings.insert(
        "true_ids".into(),
        Value::Array(true_ids.into_iter().map(Value::from).collect()),
    );
    plan.settings.insert(
        "false_ids".into(),
        Value::Array(false_ids.into_iter().map(Value::from).collect()),
    );
    Ok(())
}

/// Whether a plan classifies documents or joined rows.
pub fn has_label(found: &Operators) -> bool {
    !found.classifies.is_empty()
}

/// Build a Quail plan of one alternative's operators without the optimizer
/// rules, as `plan_region`.
///
/// # Errors
///
/// A callback fails, or the plan is malformed.
pub fn plan_region(found: &Operators, context: &PlanningContext) -> Result<Planned, CompileError> {
    let model = &context.model;
    if model.role == "reranker" {
        if has_label(found) {
            return Ok(Err(Refusal::new(
                "a reranker model cannot run AI.CLASSIFY",
                "reranker_only_scores",
                1.0,
                0.0,
                "AI.CLASSIFY expressions",
            )));
        }
        if !has_score(found) {
            return Ok(Err(Refusal::new(
                "a reranker model can only be used with AI.SCORE",
                "reranker_only_scores",
                1.0,
                0.0,
                "AI.SCORE expressions",
            )));
        }
    }
    let mut physical = match build_physical_plan(found, context)? {
        Ok(physical) => physical,
        Err(refusal) => return Ok(Err(refusal)),
    };
    if let Some(refusal) = joined_classification_refusal(&physical.nodes) {
        return Ok(Err(refusal));
    }
    bind_runtime_data(&mut physical, found, context)?;
    Ok(Ok(physical))
}

#[cfg(test)]
mod tests {
    use super::contiguous_shards;

    #[test]
    fn shards_split_documents_by_tokens_as_python_does() {
        type Case<'a> = (&'a [i64], i64, Vec<(i64, i64)>, Vec<i64>);
        let cases: [Case; 5] = [
            (
                &[5, 0, 3, 0, 0, 7, 2],
                3,
                vec![(0, 1), (1, 3), (3, 7)],
                vec![5, 3, 9],
            ),
            (
                &[1, 1, 1, 1],
                4,
                vec![(0, 1), (1, 2), (2, 3), (3, 4)],
                vec![1, 1, 1, 1],
            ),
            (&[0, 0, 4], 2, vec![(0, 3), (3, 3)], vec![4, 0]),
            (&[10, 1, 1, 1, 10], 2, vec![(0, 2), (2, 5)], vec![11, 12]),
            (&[3, 3, 3], 1, vec![(0, 3)], vec![9]),
        ];
        for (lengths, gpus, ranges, loads) in cases {
            assert_eq!(
                contiguous_shards(lengths, gpus, None),
                (ranges, loads),
                "{lengths:?}"
            );
        }
    }
}
