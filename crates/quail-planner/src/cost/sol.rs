use crate::cost::dense_decoder_cost::{
    ComponentLatency, component_latencies, dense_decoder_components,
};
use crate::cost::work::{Work, python_sum};
use crate::specs::{DeviceSpec, ModelSpec};

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

#[derive(Debug, Clone)]
pub struct SpeedOfLight {
    pub work: Work,
    pub passes: i64,
    pub components: Vec<ComponentLatency>,
}

impl SpeedOfLight {
    pub fn bytes_moved(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.component.bytes_moved.scalar().expect("summed work")),
        )
    }

    pub fn compute(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.compute_seconds.scalar().expect("summed work")),
        )
    }

    pub fn memory(&self) -> f64 {
        python_sum(
            self.components
                .iter()
                .map(|latency| latency.memory_seconds.scalar().expect("summed work")),
        )
    }

    pub fn seconds(&self) -> f64 {
        python_sum(self.components.iter().map(ComponentLatency::seconds))
    }
}

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
