//! Speed-of-light time: each model component's time at the device's
//! limits, and their sum.

use crate::cost::dense_decoder_cost::dense_decoder_components;
use crate::cost::work::{Quantity, Work, python_sum};
use crate::specs::{DeviceSpec, ModelSpec, Precision};

fn latencies(
    work: &Work,
    model: &ModelSpec,
    device: &DeviceSpec,
    passes: f64,
) -> Vec<ComponentLatency> {
    component_latencies(dense_decoder_components(work, model, passes), device)
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
            dense_decoder_components(work, model, 0.0)
                .iter()
                .map(|component| {
                    component.flops.scalar().expect("summed work")
                        / device.arithmetic_bandwidth(component.precision)
                }),
        );
    }
    python_sum(
        latencies(work, model, device, 0.0)
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
) -> f64 {
    assert!(chunk_tokens >= 1, "chunk_tokens must be at least 1");
    let passes = scalar_tokens(work) / chunk_tokens as f64;
    python_sum(
        latencies(work, model, device, passes)
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

/// Price aggregate query work with ideal packing.
///
/// # Panics
///
/// `chunk_tokens` is below 1.
pub fn speed_of_light(
    work: &Work,
    model: &ModelSpec,
    device: &DeviceSpec,
    chunk_tokens: i64,
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
        components: latencies(work, model, device, passes as f64),
    }
}

/// One model component's arithmetic and memory work.
#[derive(Debug, Clone, PartialEq)]
pub struct CostComponent {
    pub name: &'static str,
    pub flops: Quantity,
    pub bytes_moved: Quantity,
    pub precision: Precision,
    pub gemm: bool,
}

/// A component's compute and memory limits.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentLatency {
    pub component: CostComponent,
    pub compute_seconds: Quantity,
    pub memory_seconds: Quantity,
}

impl ComponentLatency {
    /// The slower of the two limits.
    ///
    /// # Panics
    ///
    /// A limit is a per-document array, which Python's `max` refuses.
    pub fn seconds(&self) -> f64 {
        let compute = self.compute_seconds.scalar().expect("scalar compute time");
        let memory = self.memory_seconds.scalar().expect("scalar memory time");
        if memory > compute { memory } else { compute }
    }
}

/// Price one component against the device's limits.
pub fn component_latency(component: CostComponent, device: &DeviceSpec) -> ComponentLatency {
    let bandwidth = if component.gemm {
        device.gemm_flops(component.precision)
    } else {
        device.arithmetic_bandwidth(component.precision)
    };
    ComponentLatency {
        compute_seconds: &component.flops / bandwidth,
        memory_seconds: &component.bytes_moved / device.hbm_bw,
        component,
    }
}

/// Price model components in execution order.
pub fn component_latencies(
    components: Vec<CostComponent>,
    device: &DeviceSpec,
) -> Vec<ComponentLatency> {
    components
        .into_iter()
        .map(|component| component_latency(component, device))
        .collect()
}
