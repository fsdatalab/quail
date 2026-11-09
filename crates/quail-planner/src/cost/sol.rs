//! Speed-of-light time.

use crate::cost::dense_decoder_cost::dense_decoder_components;
use crate::cost::quantity::python_sum;
use crate::cost::roofline::{ComponentLatency, component_latencies};
use crate::cost::specs::{DeviceSpec, ModelSpec};
use crate::cost::work::Work;

fn latencies(
    work: &Work,
    model: &ModelSpec,
    device: &DeviceSpec,
    passes: f64,
    rows: f64,
    head_rows: i64,
) -> Vec<ComponentLatency> {
    component_latencies(
        dense_decoder_components(work, model, passes, rows, head_rows),
        device,
    )
}

fn scalar_tokens(work: &Work) -> f64 {
    work.tokens
        .scalar()
        .expect("speed of light takes summed work")
}

/// Compute time without memory movement, optionally at the spec peak.
pub fn compute_seconds(work: &Work, model: &ModelSpec, device: &DeviceSpec, peak: bool) -> f64 {
    if peak {
        return python_sum(
            dense_decoder_components(work, model, 0.0, 0.0, 0)
                .iter()
                .map(|component| {
                    component.flops.scalar().expect("summed work")
                        / device.arithmetic_bandwidth(component.precision)
                }),
        );
    }
    python_sum(
        latencies(work, model, device, 0.0, 0.0, 0)
            .iter()
            .map(|latency| latency.compute_seconds.scalar().expect("summed work")),
    )
}

/// Component time with fractional ideal forward passes.
///
/// # Panics
///
/// `chunk_tokens` is below 1.
pub fn unrounded_seconds(
    work: &Work,
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    rows: f64,
    head_rows: i64,
) -> f64 {
    assert!(chunk_tokens >= 1, "chunk_tokens must be at least 1");
    let passes = scalar_tokens(work) / chunk_tokens as f64;
    python_sum(
        latencies(work, model, device, passes, rows, head_rows)
            .iter()
            .map(ComponentLatency::seconds),
    )
}

/// The speed-of-light estimate and its components.
#[derive(Debug, Clone)]
pub struct SpeedOfLight {
    pub work: Work,
    pub passes: i64,
    pub components: Vec<ComponentLatency>,
}

impl SpeedOfLight {
    /// Bytes moved by every component.
    pub fn bytes_moved(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.component.bytes_moved.scalar().expect("summed work")),
        )
    }

    /// Compute time over the components.
    pub fn compute(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.compute_seconds.scalar().expect("summed work")),
        )
    }

    /// Memory time over the components.
    pub fn memory(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.memory_seconds.scalar().expect("summed work")),
        )
    }

    /// The estimate: each component's slower limit, summed.
    pub fn seconds(&self) -> f64 {
        python_sum(self.components.iter().map(ComponentLatency::seconds))
    }
}

/// Price aggregate query work and its answer rows with ideal packing.
///
/// # Panics
///
/// `chunk_tokens` is below 1.
pub fn speed_of_light(
    work: &Work,
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
    rows: f64,
    min_passes: i64,
) -> SpeedOfLight {
    assert!(chunk_tokens >= 1, "chunk_tokens must be at least 1");
    let tokens = scalar_tokens(work);
    let passes = if tokens != 0.0 {
        (tokens / chunk_tokens as f64).ceil() as i64
    } else {
        0
    };
    let passes = passes.max(min_passes);
    SpeedOfLight {
        work: work.clone(),
        passes,
        components: latencies(work, model, device, passes as f64, rows, 0),
    }
}
