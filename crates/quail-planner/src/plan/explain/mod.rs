mod tree;
mod values;

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::Value;

use crate::operators::Operators;
use crate::plan::stages::Stage;

use tree::{Graph, physical_tree, stage_lines};
use values::{Shown, commas, commas_int, field_lines, number, py_str, seconds, width};

#[derive(Debug, Deserialize)]
pub struct ExplainRequest {
    pub logical: Option<String>,
    pub operators: Option<Operators>,
    pub physical: Option<PhysicalRequest>,
    pub refusal: Option<Value>,
    #[serde(default)]
    pub verbose: bool,
    pub rules: Option<Vec<(String, String)>>,
    pub usd_per_hour: Option<f64>,
    pub model_spec: Option<Value>,
    pub run: Option<RunRequest>,
    #[serde(default)]
    pub tree_only: bool,
}

#[derive(Debug, Deserialize)]
pub struct PhysicalRequest {
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub gpus: i64,
    #[serde(default)]
    pub settings: Vec<(String, Value)>,
    #[serde(default)]
    pub shown_settings: Vec<(String, Shown)>,
    pub graph: Value,
    #[serde(default)]
    pub estimates: Vec<(String, Value)>,
}

#[derive(Debug)]
pub struct ExtensionNode {
    pub type_name: String,
    pub inputs: Vec<(String, String, String)>,
    pub outputs: Vec<String>,
    pub fields: Vec<(String, Shown)>,
}

#[derive(Debug, Deserialize)]
pub struct RunRequest {
    pub report: Value,
    #[serde(default)]
    pub metrics: HashMap<String, Vec<(String, Value)>>,
}

pub fn explain_text(request: &ExplainRequest) -> Result<String, String> {
    if request.tree_only {
        let Some(physical) = &request.physical else {
            return Ok("no physical plan was executed".to_string());
        };
        let graph = Graph::read(physical)?;
        return Ok(physical_tree(
            &graph,
            None,
            request.verbose,
            request.run.as_ref(),
            &[],
        ));
    }
    let mut lines = vec!["logical:".to_string()];
    if let Some(logical) = &request.logical {
        lines.extend(logical.lines().map(|line| format!("  {line}")));
    }
    if let Some(refusal) = &request.refusal {
        lines.push(format!(
            "refusal: {}: needed {} {}, available {}",
            py_str(&refusal["constraint"]),
            py_str(&refusal["needed"]),
            py_str(&refusal["unit"]),
            py_str(&refusal["available"]),
        ));
        if let Some(reasons) = refusal["reasons"].as_array() {
            lines.extend(reasons.iter().map(|reason| format!("  {}", py_str(reason))));
        }
        return Ok(lines.join("\n"));
    }
    let Some(physical) = &request.physical else {
        return Ok(lines.join("\n"));
    };
    let graph = Graph::read(physical)?;
    let settings: BTreeMap<&str, &Value> = physical
        .settings
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    lines.push(String::new());
    lines.push(format!(
        "physical: backend={}, model={}, gpus={}",
        physical.backend, physical.model, physical.gpus
    ));
    if physical.backend == "quail" {
        let mut budgets = vec!["KV=bf16".to_string()];
        if let Some(chunk) = settings
            .get("chunk_tokens")
            .filter(|value| !value.is_null())
        {
            budgets.push(format!("chunk budget={} tokens", commas(chunk)));
        }
        if let Some(admission) = settings
            .get("admission_tokens")
            .filter(|value| !value.is_null())
            && reuses_kv(request.model_spec.as_ref())
        {
            budgets.push(format!("admission budget={} tokens", commas(admission)));
        }
        lines.push(format!("  {}", budgets.join(", ")));
        if let Some(Value::Object(deduplicate)) = settings.get("deduplicate") {
            for (alias, columns) in deduplicate {
                let mut grouping = vec!["text".to_string()];
                grouping.extend(columns.as_array().into_iter().flatten().map(py_str));
                lines.push(format!(
                    "  deduplicate: {alias} (by {})",
                    grouping.join(" and ")
                ));
            }
        }
        lines.extend(
            architecture_lines(request.model_spec.as_ref())
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    lines.push(String::new());
    let tree = physical_tree(
        &graph,
        request.operators.as_ref(),
        request.verbose,
        request.run.as_ref(),
        &physical.estimates,
    );
    lines.extend(tree.lines().map(|line| format!("  {line}")));
    if physical.gpus > 1
        && let Some(cut) = settings.get("stages")
    {
        let cut: Vec<Stage> =
            serde_json::from_value((*cut).clone()).map_err(|error| error.to_string())?;
        lines.extend(
            stage_lines(&cut)
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    let passes = physical
        .estimates
        .iter()
        .any(|(_, estimate)| estimate.get("pass_members").is_some());
    if passes {
        lines.push(
            "  est. time is each pass group's work, on its first operator; the other \
             operators in the group are asked in its passes"
                .to_string(),
        );
    } else if !physical.estimates.is_empty() {
        lines.push(
            "  est. time is each node's work alone; node times do not add up to the plan \
             estimate"
                .to_string(),
        );
        lines.push("  because chunk packing shares forward passes across nodes".to_string());
    }
    if let Some(run) = &request.run {
        lines.push(String::new());
        lines.push("run:".to_string());
        lines.extend(
            run_summary(&run.report, &graph, physical.gpus, request.usd_per_hour)
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    if request.verbose {
        lines.push(String::new());
        lines.push("settings:".to_string());
        let mut fields = vec![
            ("model".to_string(), Shown::Text(physical.model.clone())),
            ("device".to_string(), Shown::Text(physical.device.clone())),
        ];
        fields.extend(
            physical
                .shown_settings
                .iter()
                .filter(|(name, _)| name != "stages")
                .cloned(),
        );
        lines.extend(field_lines(&fields, 1));
        if physical.backend == "quail" {
            lines.push("  KV dtype=bf16".to_string());
        }
        if let Some(rules) = &request.rules {
            lines.push(String::new());
            lines.extend(rule_lines(rules));
        }
    }
    Ok(lines.join("\n"))
}

fn run_summary(report: &Value, graph: &Graph, gpus: i64, usd_per_hour: Option<f64>) -> Vec<String> {
    let Some(wall) = report.get("wall_s").and_then(Value::as_f64) else {
        return vec!["query time   not measured".to_string()];
    };
    let mut items = vec![(
        "query time",
        format!("{} (model startup excluded)", seconds(Some(wall))),
    )];
    if let Some(boot) = report.get("boot_s").filter(|boot| !boot.is_null()) {
        let kind = report
            .get("boot_kind")
            .filter(|kind| kind.as_str().is_some_and(|kind| !kind.is_empty()))
            .map(|kind| format!(" ({})", py_str(kind)))
            .unwrap_or_default();
        items.push(("startup", seconds(boot.as_f64()) + &kind));
    }
    let mut pairs: f64 = report["stages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|stage| stage["op"].as_str() == Some("join"))
        .map(|stage| stage.get("tuples").and_then(Value::as_f64).unwrap_or(0.0))
        .sum();
    if pairs == 0.0 {
        pairs = report
            .get("evaluated_document_pairs")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
    }
    let documents = graph.num_documents();
    if wall > 0.0 && pairs != 0.0 {
        items.push((
            "throughput",
            format!(
                "{} document pairs/second over {} evaluated pairs",
                number(pairs / wall),
                commas_int(pairs as i64)
            ),
        ));
    } else if wall > 0.0 {
        items.push((
            "throughput",
            format!(
                "{} documents/second over {} input documents",
                number(documents as f64 / wall),
                commas_int(documents)
            ),
        ));
    }
    let mut tokens = format!("{} fresh", commas(&report["fresh_tokens"]));
    if let Some(cached) = report
        .get("cached_tokens")
        .filter(|cached| !cached.is_null())
    {
        tokens += &format!(", {} read from KV", commas(cached));
    }
    items.push(("tokens", tokens));
    if let Some(price) = usd_per_hour {
        let cost = wall / 3600.0 * gpus as f64 * price;
        items.push((
            "GPU cost",
            format!("${cost:.4} per query ({gpus} GPU at ${price:.4}/hour, startup excluded)"),
        ));
    }
    let label_width = items
        .iter()
        .map(|(label, _)| width(label))
        .max()
        .unwrap_or(0);
    items
        .into_iter()
        .map(|(label, value)| format!("{label:<label_width$}   {value}"))
        .collect()
}

fn rule_lines(rules: &[(String, String)]) -> Vec<String> {
    let names = |kind: &str| {
        let names: Vec<&str> = rules
            .iter()
            .filter(|(_, found)| found == kind)
            .map(|(name, _)| name.as_str())
            .collect();
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    };
    vec![
        "optimizer rules:".to_string(),
        format!("  logical: {}", names("logical")),
        format!("  physical: {}", names("physical")),
        "  search: rewrite logical candidates, then lower each alternative".to_string(),
        format!("  refine each graph in order: {}", names("physical")),
        "  selection: lowest estimated cost after graph refinement".to_string(),
    ]
}

fn reuses_kv(spec: Option<&Value>) -> bool {
    spec.is_none_or(|spec| spec["attention_mask"].as_str() != Some("set"))
}

fn architecture_lines(spec: Option<&Value>) -> Vec<String> {
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut kinds: Vec<(String, usize)> = Vec::new();
    for kind in spec["layer_kinds"].as_array().into_iter().flatten() {
        let kind = py_str(kind);
        match kinds.iter_mut().find(|(found, _)| *found == kind) {
            Some((_, count)) => *count += 1,
            None => kinds.push((kind, 1)),
        }
    }
    let has = |name: &str| kinds.iter().any(|(kind, _)| kind == name);
    let layers = if kinds.is_empty() {
        format!("{} layers", py_str(&spec["layers"]))
    } else {
        kinds
            .iter()
            .map(|(kind, count)| format!("{count} {kind}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let precision = py_str(&spec["weight_precision"]);
    let mut gemm = if precision == "fp8" {
        "fp8 GEMM (DeepGEMM)".to_string()
    } else {
        "bf16 GEMM (cuBLAS)".to_string()
    };
    let arch = py_str(&spec["arch"]);
    if arch == "qwen3_5" {
        gemm = "vLLM per-layer GEMM".to_string();
        if let Some(recurrent) = spec
            .get("gdn_weight_precision")
            .filter(|value| !value.is_null())
        {
            gemm += &format!(" (recurrent projections {})", py_str(recurrent));
        }
        if let Some(mlp) = spec
            .get("mlp_weight_precision")
            .filter(|value| !value.is_null())
        {
            gemm += &format!(" (MLP projections {})", py_str(mlp));
        }
    }
    let mut paths = vec![format!("linear layers: {gemm}")];
    if spec["attention_mask"].as_str() == Some("set") {
        paths.push(
            "attention: FlashAttention, set attention on the full layers as three calls \
             merged by log-sum-exp, no KV kept"
                .to_string(),
        );
    } else if has("full_attention") || has("sliding_attention") {
        paths.push("attention: FlashAttention over paged KV".to_string());
    }
    if has("gated_delta_net") {
        let requested = std::env::var("QUAIL_GDN_PREFILL").unwrap_or_else(|_| "auto".into());
        paths.push(format!(
            "gated_delta_net: Triton convolution, recurrence={requested} (QUAIL_GDN_PREFILL)"
        ));
    }
    if has("short_conv") {
        paths.push("short_conv: Triton convolution".to_string());
    }
    vec![
        format!("architecture={arch} ({layers}), weights={precision}"),
        format!("kernels: {}", paths.join("; ")),
    ]
}
