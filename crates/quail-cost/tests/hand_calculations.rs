//! The cost model's work counts and latencies against hand calculations
//! and brute-force attention masks.

use std::collections::BTreeMap;
use std::path::Path;

use quail_cost::dense_decoder_cost::{dense_decoder_components, dense_params};
use quail_cost::quantity::Quantity;
use quail_cost::sol::speed_of_light;
use quail_cost::specs::{DeviceSpec, ModelSpec, read_devices, read_models};
use quail_cost::work::{Work, ask, scan, stream};

fn data() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../quail/specs/data"
    ))
}

fn models() -> BTreeMap<String, ModelSpec> {
    read_models(&data().join("models")).expect("model specs")
}

fn h100() -> DeviceSpec {
    read_devices(&data().join("devices")).expect("device specs")["h100-sxm"].clone()
}

fn value(quantity: &Quantity) -> f64 {
    quantity.total()
}

fn q(value: f64) -> Quantity {
    Quantity::Scalar(value)
}

/// Allowed keys for every new query position, counted one by one.
fn mask_pairs(prefix: i64, suffix: i64, window: i64) -> f64 {
    (prefix..prefix + suffix)
        .map(|query| {
            (0..=query)
                .filter(|key| window == 0 || query - key < window)
                .count()
        })
        .sum::<usize>() as f64
}

#[test]
fn work_counts_the_pairs_the_attention_mask_allows() {
    for (window, prefix, suffix) in [
        (0, 2, 4),
        (1, 9, 3),
        (4, 0, 3),
        (4, 2, 4),
        (4, 4, 1),
        (4, 9, 3),
    ] {
        let (p, s) = (prefix as f64, suffix as f64);
        let first = scan(&q(p), &q(s), window);
        let continuation = ask(&q(p), &q(s), window);
        assert_eq!(value(&first.pairs), mask_pairs(0, prefix + suffix, 0));
        assert_eq!(value(&continuation.pairs), mask_pairs(prefix, suffix, 0));
        let sliding = |pairs: f64| if window == 0 { 0.0 } else { pairs };
        assert_eq!(
            value(&first.sliding_pairs),
            sliding(mask_pairs(0, prefix + suffix, window))
        );
        assert_eq!(
            value(&continuation.sliding_pairs),
            sliding(mask_pairs(prefix, suffix, window))
        );
        assert_eq!(
            value(&continuation.sliding_kv_read),
            if window == 0 {
                0.0
            } else {
                p.min((window - 1) as f64)
            }
        );
        let branches = stream(&q(p), &[q(s), q(s + 1.0)], window);
        assert_eq!(
            value(&branches.sliding_pairs),
            sliding(mask_pairs(prefix, suffix, window) + mask_pairs(prefix, suffix + 1, window))
        );
        assert_eq!(
            value(&branches.sliding_kv_read),
            value(&continuation.sliding_kv_read)
        );
    }
}

#[test]
fn qwen3_4b_reads_each_dense_weight_once_per_pass() {
    let models = models();
    let model = &models["qwen3-4b-fp8"];
    let work = &ask(&q(400.0), &q(50.0), 0) * 1000.0;
    let result = speed_of_light(&work, model, &h100(), 110_376, 0.0, 0);
    let names: Vec<&str> = result
        .components
        .iter()
        .map(|component| component.component.name)
        .collect();
    assert_eq!(names, ["attn_proj", "mlp", "attention"]);
    let slower: f64 = result
        .components
        .iter()
        .map(|component| component.seconds())
        .sum();
    assert!((result.seconds() - slower).abs() <= 1e-12 * slower);
    assert!(result.seconds() > result.compute().max(result.memory()));
    let dense_bytes: f64 = result
        .components
        .iter()
        .filter(|component| component.component.name != "attention")
        .map(|component| value(&component.component.bytes_moved))
        .sum();
    assert_eq!(dense_bytes, (dense_params(model) * result.passes) as f64);
    let mut larger_copy = model.clone();
    larger_copy.w_mem_bytes = 150e9;
    let larger = speed_of_light(&work, &larger_copy, &h100(), 110_376, 0.0, 0);
    assert!((larger.seconds() - result.seconds()).abs() <= 1e-12 * result.seconds());
}

#[test]
fn diffusion_gemma_attention_reads_full_and_sliding_kv() {
    let models = models();
    let model = &models["diffusion-gemma-26b-a4b-fp8"];
    let work = ask(&q(4096.0), &q(16.0), model.sliding_window);
    let attention = &dense_decoder_components(&work, model, 1.0, 0.0, 0)[2];
    assert_eq!(
        value(&attention.flops),
        4.0 * 16.0 * (5.0 * 512.0 * value(&work.pairs) + 25.0 * 256.0 * value(&work.sliding_pairs))
    );
    assert_eq!(
        value(&attention.bytes_moved),
        (5 * 2 * 2 * 512 * 2 * (16 + 4096) + 25 * 2 * 8 * 256 * 2 * (16 + 1023)) as f64
    );
}

#[test]
fn one_review_through_a_filter_and_a_join_matches_the_hand_calculation() {
    let models = models();
    let model = &models["qwen3.5-4b-bf16"];
    // 100-token review, 6-token preamble, 56-token filter question, then a
    // join: a 43-token frame and two aspects of 2 and 3 tokens, each with a
    // 13-token tail
    let checkpoint = Work {
        state_written: q(1.0),
        ..Work::default()
    };
    let work = &(&(&scan(&q(106.0), &q(56.0), 0) + &ask(&q(106.0), &q(43.0), 0)) + &checkpoint)
        + &stream(&q(149.0), &[q(15.0), q(16.0)], 0);
    assert_eq!(value(&work.tokens), 236.0);
    assert_eq!(value(&work.pairs), (13_203 + 5_504 + 4_875) as f64);
    assert_eq!(value(&work.kv_written), 236.0);
    assert_eq!(value(&work.kv_read), (106 + 149) as f64);
    assert_eq!(
        (value(&work.state_written), value(&work.state_read)),
        (2.0, 3.0)
    );
    let sol = speed_of_light(&work, model, &h100(), 116_508, 0.0, 0);
    assert_eq!(sol.passes, 1);
    let names: Vec<&str> = sol
        .components
        .iter()
        .map(|component| component.component.name)
        .collect();
    assert_eq!(
        names,
        ["attn_proj", "mlp", "attention", "gdn_core", "state", "conv"]
    );
    let component = |name: &str| {
        sol.components
            .iter()
            .find(|component| component.component.name == name)
            .expect("component")
    };
    // one pass reads every weight once, so the weight components are
    // limited by memory at this size
    let projection = component("attn_proj");
    assert!(value(&projection.memory_seconds) > value(&projection.compute_seconds));
    assert_eq!(
        value(&component("state").component.bytes_moved),
        5.0 * 51_511_296.0
    );
    assert!((sol.seconds() / 2.2516e-3 - 1.0).abs() < 1e-3);
}
