//! Each model component's time at the device's limits.

use crate::cost::quantity::Quantity;
use crate::cost::specs::{DeviceSpec, Precision};

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
