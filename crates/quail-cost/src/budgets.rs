//! Memory budgets and attention-path choice, as
//! `quail/common/budgets.py` computes them.
//!
//! Python mixes integers and floats here; each function keeps the type
//! Python's result has, and `//` on floats rounds as CPython's does.

use crate::specs::{DeviceSpec, ModelSpec};

/// Fraction of device memory the executor claims.
pub const POOL_FRACTION: f64 = 0.95;
/// Default KV page size; devices may require larger pages.
pub const PAGE_TOKENS: i64 = 16;
pub const INT32_MAX: i64 = (1 << 31) - 1;
/// bf16 activations, bytes per element.
pub const ACT_BYTES: i64 = 2;
/// Chunks of activation memory reserved outside the arena.
pub const ACT_RESERVE_CHUNKS: i64 = 2;
pub const READOUT_BLOCK_ROWS: i64 = 512;
/// Questions one pass may start from checkpoints at once.
pub const CHECKPOINT_PASS_QUESTIONS: i64 = 1_024;
/// Document length a hybrid model's retained pool is split for when the
/// mean is unknown.
pub const CHECKPOINT_DEFAULT_DOC_TOKENS: i64 = 1_000;
pub const MIB: i64 = 1 << 20;
pub const INTERMEDIATE_BLOCK_TOKENS: i64 = 64;
/// Rows past a document that its pages also cover.
pub const SPLIT_EXTRA_TOKENS: i64 = 64;
/// Below this many dense parameters a model runs unified attention only.
pub const TREE_ATTENTION_MIN_PARAMS: f64 = 2e9;
/// Families whose pipelines split their forward pass into captured pieces.
pub const GRAPH_ARCHS: [&str; 2] = ["qwen3", "qwen3_5"];

/// CUDA graph capture sizes, in tokens per pass.
pub fn graph_token_buckets_all() -> Vec<i64> {
    let mut sizes = vec![1, 2, 4];
    sizes.extend((8..256).step_by(8));
    sizes.extend((256..513).step_by(16));
    sizes.extend((768..8_193).step_by(256));
    sizes
}

/// CPython's `float // float` (`float_floor_div`).
pub fn floor_divide(a: f64, b: f64) -> f64 {
    let remainder = a % b;
    let mut quotient = (a - remainder) / b;
    if remainder != 0.0 && ((b < 0.0) != (remainder < 0.0)) {
        quotient -= 1.0;
    }
    if quotient != 0.0 {
        let floor = quotient.floor();
        if quotient - floor > 0.5 {
            floor + 1.0
        } else {
            floor
        }
    } else {
        0.0_f64.copysign(a / b)
    }
}

/// Python's `int // int`, which rounds toward negative infinity.
pub fn floor_divide_int(a: i64, b: i64) -> i64 {
    let quotient = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        quotient - 1
    } else {
        quotient
    }
}

/// KV pages that hold `tokens`: `-(-tokens // page_tokens)`.
pub fn pages(tokens: i64, page_tokens: i64) -> i64 {
    -floor_divide_int(-tokens, page_tokens)
}

/// `pages` of a float token count, which Python returns as a float.
pub fn pages_of(tokens: f64, page_tokens: i64) -> f64 {
    -floor_divide(-tokens, page_tokens as f64)
}

/// Readout workspace shared with copy verification, capped by one
/// activation chunk.
pub fn copy_verification_workspace_bytes(model: &ModelSpec, chunk_tokens: i64) -> i64 {
    let activations = chunk_tokens as f64 * model.act_per_token();
    let logits = 2 * READOUT_BLOCK_ROWS * model.vocab * 4;
    if (logits as f64) < activations {
        logits
    } else {
        activations as i64
    }
}

/// How many pooled GPU memories would hold the loaded weights.
pub fn minimum_weight_gpus(model: &ModelSpec, device: &DeviceSpec) -> i64 {
    let mut gpus = 1;
    while model.W_mem() > device.mem_bytes * POOL_FRACTION * gpus as f64 {
        gpus *= 2;
    }
    gpus
}

/// Max tokens per chunk from the int32 element-offset limit.
pub fn kernel_index_cap(model: &ModelSpec) -> i64 {
    let experts = model.experts_active * model.hidden.max(2 * model.expert_intermediate);
    let width = model.widest_projection().max(experts);
    floor_divide_int(INT32_MAX, width)
}

/// CUDA graph capture sizes of a dense Qwen3 or Qwen3.5 model.
pub fn graph_token_buckets(model: &ModelSpec) -> Vec<i64> {
    if !GRAPH_ARCHS.contains(&model.arch.as_str())
        || model.experts != 0
        || model.sae.is_some()
        || model.canvas_tokens != 0
    {
        return Vec::new();
    }
    let cap = kernel_index_cap(model);
    graph_token_buckets_all()
        .into_iter()
        .filter(|size| *size <= cap)
        .collect()
}

/// Bytes of the shared CUDA graph boundary buffers and workspace.
pub fn graph_memory_bytes(model: &ModelSpec) -> i64 {
    match graph_token_buckets(model).last() {
        None => 0,
        Some(largest) => ((2 * largest) as f64 * model.act_per_token()).ceil() as i64,
    }
}

/// Bytes left for KV after weights, activations, and CUDA graph storage.
pub fn arena_bytes(model: &ModelSpec, device: &DeviceSpec, chunk_tokens: i64) -> f64 {
    device.mem_bytes * POOL_FRACTION
        - model.W_resident()
        - (ACT_RESERVE_CHUNKS * chunk_tokens) as f64 * model.act_per_token()
        - graph_memory_bytes(model) as f64
}

/// Sliding-layer pages one chunk's fresh prefixes take before trimming.
pub fn transient_sliding_pages(chunk_tokens: i64, window: i64, page_tokens: i64) -> i64 {
    pages(chunk_tokens, page_tokens) + -floor_divide_int(-chunk_tokens, window)
}

/// KV pages of memory one document checkpoint takes.
pub fn checkpoint_pages(model: &ModelSpec, page_tokens: i64) -> f64 {
    model.checkpoint_bytes() as f64 / (model.kappa() * page_tokens as f64)
}

/// Temporary bytes a pass of `questions` starts from checkpoints.
pub fn checkpoint_pass_bytes(model: &ModelSpec, questions: i64, chunk_tokens: i64) -> i64 {
    let buffers = questions + -floor_divide_int(-chunk_tokens, INTERMEDIATE_BLOCK_TOKENS);
    questions * (model.gdn_state_bytes() + model.conv_state_bytes()) + buffers * MIB
}

/// KV pages kept free for two chunks of execution.
pub fn execution_pages(chunk_tokens: i64, page_tokens: i64) -> i64 {
    2 * pages(chunk_tokens, page_tokens)
}

/// Workspace, execution KV, and one retained document's KV and state.
pub fn checkpoint_memory_bytes(model: &ModelSpec, chunk_tokens: i64, page_tokens: i64) -> f64 {
    if model.checkpoint_bytes() == 0 || !model.reuses_kv() {
        return 0.0;
    }
    let document_tokens = chunk_tokens.max(CHECKPOINT_DEFAULT_DOC_TOKENS);
    let reserved = execution_pages(chunk_tokens, page_tokens)
        + pages(document_tokens, page_tokens)
        + checkpoint_pages(model, page_tokens).ceil() as i64;
    checkpoint_pass_bytes(model, CHECKPOINT_PASS_QUESTIONS, chunk_tokens) as f64
        + reserved as f64 * model.kappa() * page_tokens as f64
}

/// Resident weights, activation memory, and checkpoint reservations.
pub fn chunk_memory_bytes(model: &ModelSpec, chunk_tokens: i64, page_tokens: i64) -> f64 {
    model.W_resident()
        + (ACT_RESERVE_CHUNKS * chunk_tokens) as f64 * model.act_per_token()
        + checkpoint_memory_bytes(model, chunk_tokens, page_tokens)
}

/// The largest chunk that fits its memory reservations, or zero.
pub fn chunk_memory_bound(model: &ModelSpec, device: &DeviceSpec) -> i64 {
    let available = device.mem_bytes * POOL_FRACTION - graph_memory_bytes(model) as f64;
    let free = available - model.W_resident();
    let mut lower = 0;
    let reserve = ACT_RESERVE_CHUNKS as f64 * model.act_per_token();
    let mut upper = 0_i64.max(floor_divide(free, reserve) as i64);
    while lower < upper {
        let chunk = floor_divide_int(lower + upper + 1, 2);
        if chunk_memory_bytes(model, chunk, device.page_tokens) <= available {
            lower = chunk;
        } else {
            upper = chunk - 1;
        }
    }
    lower
}

/// Pages a hybrid model shares between retained KV and checkpoints.
pub fn hybrid_pool_pages(model: &ModelSpec, device: &DeviceSpec, chunk_tokens: i64) -> i64 {
    let free = arena_bytes(model, device, chunk_tokens)
        - checkpoint_pass_bytes(model, CHECKPOINT_PASS_QUESTIONS, chunk_tokens) as f64;
    let page_bytes = model.kappa() * device.page_tokens as f64;
    floor_divide(free, page_bytes) as i64 - execution_pages(chunk_tokens, device.page_tokens)
}

/// Checkpoints the pool holds beside an arena of `kv_pages`; 0 without them.
pub fn checkpoint_slots(
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    kv_pages: i64,
) -> i64 {
    let checkpoint = checkpoint_pages(model, device.page_tokens);
    if checkpoint == 0.0 {
        return 0;
    }
    let retained = kv_pages - execution_pages(chunk_tokens, device.page_tokens);
    let pool = hybrid_pool_pages(model, device, chunk_tokens);
    0_i64.max(floor_divide((pool - retained) as f64, checkpoint) as i64 - 1)
}

/// Arena KV pages of a hybrid model: two chunks of execution plus the
/// retained KV.
///
/// # Errors
///
/// The pool cannot hold a chunk's KV, or one checkpoint beside a document.
pub fn hybrid_arena_pages(
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    mean_document_tokens: Option<f64>,
) -> Result<i64, String> {
    let pool = hybrid_pool_pages(model, device, chunk_tokens);
    let checkpoint = checkpoint_pages(model, device.page_tokens);
    let (mean, document_pages) = match mean_document_tokens {
        None => (
            CHECKPOINT_DEFAULT_DOC_TOKENS as f64,
            pages(CHECKPOINT_DEFAULT_DOC_TOKENS, device.page_tokens) as f64,
        ),
        Some(mean) => {
            let mean = if mean > 1.0 { mean } else { 1.0 };
            (mean, pages_of(mean, device.page_tokens))
        }
    };
    let slots = floor_divide(pool as f64, document_pages + checkpoint) as i64;
    let kv_pages = (pool as f64 - slots as f64 * checkpoint) as i64;
    if kv_pages <= 0 {
        return Err("the pool cannot hold one chunk's KV beside a checkpoint".to_string());
    }
    if slots < 1 {
        return Err(format!(
            "the pool holds no checkpoint of {checkpoint:.1} pages beside a document of \
             {mean:.0} tokens"
        ));
    }
    Ok(kv_pages + execution_pages(chunk_tokens, device.page_tokens))
}

/// Pages of the two KV pools: (every-token pool, sliding-layer pool).
///
/// # Errors
///
/// The arena cannot hold one chunk's KV.
pub fn arena_pages(
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: Option<i64>,
    mean_document_tokens: Option<f64>,
) -> Result<(i64, i64), String> {
    if !model.reuses_kv() {
        return Ok((
            1,
            if model.sliding_layer_set().is_empty() {
                0
            } else {
                1
            },
        ));
    }
    let chunk_tokens = match chunk_tokens {
        Some(chunk) => chunk,
        None => chunk_budget(model, device),
    };
    if !model.state_layers().is_empty() {
        return Ok((
            hybrid_arena_pages(model, device, chunk_tokens, mean_document_tokens)?,
            0,
        ));
    }
    let free = arena_bytes(model, device, chunk_tokens);
    let page_bytes_full = model.kappa_full() * device.page_tokens as f64;
    let page_bytes_sliding = model.kappa_sliding() * device.page_tokens as f64;
    if page_bytes_sliding == 0.0 {
        return Ok((floor_divide(free, page_bytes_full) as i64, 0));
    }
    let window = model.sliding_window;
    let mean = match mean_document_tokens {
        None => window as f64,
        Some(mean) => {
            if mean > 1.0 {
                mean
            } else {
                1.0
            }
        }
    };
    let visible = if (window as f64) < mean {
        window as f64
    } else {
        mean
    };
    let fraction = (visible + SPLIT_EXTRA_TOKENS as f64) / (mean + SPLIT_EXTRA_TOKENS as f64);
    let ratio = if fraction < 1.0 { fraction } else { 1.0 };
    let mut full = floor_divide(free, page_bytes_full + ratio * page_bytes_sliding) as i64;
    let mut sliding = (full as f64 * ratio) as i64;
    let floor = transient_sliding_pages(chunk_tokens, window, device.page_tokens);
    if sliding < floor {
        sliding = floor;
        full = floor_divide(free - sliding as f64 * page_bytes_sliding, page_bytes_full) as i64;
    }
    if full <= 0 {
        return Err("the KV arena cannot hold one chunk's sliding KV".to_string());
    }
    Ok((full, sliding))
}

/// Admission budget: tokens of document KV that can be resident at once.
///
/// # Errors
///
/// The arena cannot hold one chunk's KV.
pub fn arena_tokens(
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: Option<i64>,
    mean_document_tokens: Option<f64>,
) -> Result<i64, String> {
    let (full, _) = arena_pages(model, device, chunk_tokens, mean_document_tokens)?;
    Ok(full * device.page_tokens)
}

/// Whether the leftover arena holds one chunk's sliding KV.
pub fn sliding_kv_fits(model: &ModelSpec, device: &DeviceSpec, chunk_tokens: Option<i64>) -> bool {
    model.sliding_window == 0 || arena_pages(model, device, chunk_tokens, None).is_ok()
}

/// Shrink a chunk until its activation reserve leaves its own KV.
fn chunk_leaving_kv(model: &ModelSpec, device: &DeviceSpec, chunk: i64) -> i64 {
    if model.sliding_window != 0 || chunk < 1 {
        return chunk;
    }
    let free =
        device.mem_bytes * POOL_FRACTION - model.W_resident() - graph_memory_bytes(model) as f64;
    let per = ACT_RESERVE_CHUNKS as f64 * model.act_per_token() + model.kappa_full();
    let mut chunk = chunk;
    if per > 0.0 {
        chunk = chunk.min(1_i64.max(floor_divide(free, per) as i64));
    }
    let full_pages = |chunk: i64| {
        arena_pages(model, device, Some(chunk), None)
            .expect("a model without sliding layers")
            .0
    };
    let mut full = full_pages(chunk);
    while chunk > 1 && full * device.page_tokens < chunk {
        chunk -= 1;
        full = full_pages(chunk);
    }
    chunk
}

/// The chunk budget allowed by the model and device.
pub fn chunk_budget(model: &ModelSpec, device: &DeviceSpec) -> i64 {
    let mut budget = chunk_memory_bound(model, device).min(kernel_index_cap(model));
    if model.chunk_cap_tokens != 0 {
        budget = budget.min(model.chunk_cap_tokens);
    }
    chunk_leaving_kv(model, device, budget)
}

fn precision_bytes(precision: &str) -> i64 {
    match precision {
        "fp8" => 1,
        "bf16" => 2,
        "fp32" => 4,
        other => panic!("unsupported precision: {other}"),
    }
}

/// (in dim, out dim, layers, precision) of each dense projection.
fn projection_shapes(model: &ModelSpec) -> Vec<(i64, i64, i64, String)> {
    let query_width = if model.attn_output_gate { 2 } else { 1 };
    let qkv_out = (query_width * model.num_q_heads + 2 * model.num_kv_heads) * model.d_head;
    let attention = model.attention_layers().len() as i64;
    let precision = model.weight_precision.clone();
    let mut shapes = vec![
        (model.hidden, qkv_out, attention, precision.clone()),
        (
            model.num_q_heads * model.d_head,
            model.hidden,
            attention,
            precision.clone(),
        ),
        (
            model.hidden,
            model.ffn_width,
            model.layers,
            precision.clone(),
        ),
        (
            model.intermediate(),
            model.hidden,
            model.layers,
            precision.clone(),
        ),
    ];
    let gdn = model.gdn_layers().len() as i64;
    if gdn != 0 {
        let key_width = model.gdn_key_heads * model.gdn_key_dim;
        let value_width = model.gdn_value_heads * model.gdn_value_dim;
        let gdn_precision = model
            .gdn_weight_precision
            .clone()
            .unwrap_or_else(|| precision.clone());
        shapes.push((
            model.hidden,
            2 * key_width + 2 * value_width + 2 * model.gdn_value_heads,
            gdn,
            gdn_precision.clone(),
        ));
        shapes.push((value_width, model.hidden, gdn, gdn_precision));
    }
    let short = model.short_conv_layers().len() as i64;
    if short != 0 {
        shapes.push((model.hidden, 3 * model.hidden, short, precision.clone()));
        shapes.push((model.hidden, model.hidden, short, precision));
    }
    shapes
}

/// Chunk size (tokens) where the dense projections become compute-bound.
///
/// # Errors
///
/// The projections never cross the roofline ridge.
pub fn compute_knee(model: &ModelSpec, device: &DeviceSpec) -> Result<f64, String> {
    let (mut compute_seconds, mut weight_bytes, mut activation_bytes) = (0.0, 0.0, 0.0);
    for (din, dout, layers, precision) in projection_shapes(model) {
        let share = layers as f64 / model.layers as f64;
        let params = share * din as f64 * dout as f64;
        compute_seconds += 2.0 * params / device.gemm_flops(&precision);
        weight_bytes += params * precision_bytes(&precision) as f64;
        activation_bytes += share * (din + dout) as f64 * ACT_BYTES as f64;
    }
    let denominator = compute_seconds * device.hbm_bw - activation_bytes;
    if denominator <= 0.0 {
        return Err("projections never cross the ridge".to_string());
    }
    Ok(weight_bytes / denominator)
}

/// Ideal seconds for the attention kernels in one chunk, all layers.
fn attention_time(model: &ModelSpec, device: &DeviceSpec, chunk: f64, context: f64) -> f64 {
    let layers = model.attention_layers().len() as f64;
    let flops = 4.0 * chunk * context * model.num_q_heads as f64 * model.d_head as f64;
    let moved = context * model.kappa() / layers
        + 2.0 * chunk * model.num_q_heads as f64 * model.d_head as f64 * ACT_BYTES as f64;
    let compute = flops / device.arithmetic_bandwidth(&model.attention_precision);
    let memory = moved / device.hbm_bw;
    (if memory > compute { memory } else { compute }) * layers
}

/// Whether the planner and the executor may run the model's tree path.
pub fn tree_attention_allowed(model: &ModelSpec) -> bool {
    model.canvas_tokens == 0
        && model.state_layers().is_empty()
        && model.params >= TREE_ATTENTION_MIN_PARAMS
}

/// Pick "unified" or "tree" attention for readers of one shared node.
pub fn cheaper_attention_path(
    model: &ModelSpec,
    device: &DeviceSpec,
    readers: f64,
    reader_rows: f64,
    node_tokens: f64,
) -> &'static str {
    if readers <= 1.0 || reader_rows <= 0.0 || node_tokens <= 0.0 {
        return "unified";
    }
    let unified = readers * attention_time(model, device, reader_rows, node_tokens);
    let rows = readers * reader_rows;
    let mut tree = attention_time(model, device, rows, node_tokens);
    let merge = 3.0 * rows * model.num_q_heads as f64 * model.d_head as f64 * ACT_BYTES as f64
        / device.hbm_bw;
    tree += merge * model.attention_layers().len() as f64;
    if tree < unified { "tree" } else { "unified" }
}

/// The model copies that share work spread evenly over documents.
pub fn parallel_copies(gpus: i64, num_documents: f64) -> f64 {
    let fewer = if num_documents < gpus as f64 {
        num_documents
    } else {
        gpus as f64
    };
    if fewer > 1.0 { fewer } else { 1.0 }
}
