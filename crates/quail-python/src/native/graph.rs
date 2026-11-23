//! Quail's physical graph for Python: the graph that holds the nodes of
//! [`super::physical`], and the plan values Python reads and writes.

use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};
use quail_planner::physical;
use quail_planner::topology;
use serde_json::Value;

use super::physical::{Tagged, from_python, is_known_tag, node_object, to_python};

pyo3::create_exception!(
    _quail_native,
    GraphValidationError,
    PyValueError,
    "Raised when a physical graph is invalid."
);

// ----------------------------------------------------------------- the graph

/// An immutable graph of physical nodes and the port that returns the
/// result.
#[pyclass(name = "PhysicalGraph", module = "quail.physical", frozen)]
pub struct PhysicalGraph {
    nodes: Py<PyTuple>,
    root: Py<PyAny>,
    graph: physical::PhysicalGraph,
}

impl PhysicalGraph {
    fn of<'py>(py: Python<'py>, graph: physical::PhysicalGraph) -> PyResult<Bound<'py, Self>> {
        let nodes = graph
            .nodes
            .iter()
            .map(|node| node_object(py, Arc::new(node.clone())))
            .collect::<PyResult<Vec<_>>>()?;
        let root = to_python(
            py,
            &serde_json::to_value(&graph.root).expect("a port writes"),
        )?;
        Bound::new(
            py,
            Self {
                nodes: PyTuple::new(py, nodes)?.unbind(),
                root: root.unbind(),
                graph,
            },
        )
    }

    fn by_position<'py>(
        &self,
        py: Python<'py>,
        positions: &[usize],
    ) -> PyResult<Bound<'py, PyTuple>> {
        let nodes = self.nodes.bind(py);
        PyTuple::new(
            py,
            positions
                .iter()
                .map(|position| nodes.get_item(*position))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }
}

#[pymethods]
impl PhysicalGraph {
    #[new]
    fn new(nodes: &Bound<'_, PyAny>, root: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = nodes.py();
        let objects = nodes
            .try_iter()?
            .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?;
        let typed = objects
            .iter()
            .map(|node| Tagged::held_node(node).map(|node| node.as_ref().clone()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            nodes: PyTuple::new(py, objects)?.unbind(),
            root: root.clone().unbind(),
            graph: physical::PhysicalGraph {
                nodes: typed,
                root: Tagged::held_value(root)?,
            },
        })
    }

    /// Read and validate a graph written as `{"nodes", "root"}` JSON.
    #[staticmethod]
    fn from_json<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, Self>> {
        let graph: physical::PhysicalGraph =
            serde_json::from_str(text).map_err(|error| PyValueError::new_err(error.to_string()))?;
        topology::validate_graph(&graph.nodes, &graph.root)
            .map_err(GraphValidationError::new_err)?;
        Self::of(py, graph)
    }

    /// The graph as JSON, as plans write it.
    fn to_json(&self) -> String {
        serde_json::to_string(&self.graph).expect("a graph writes")
    }

    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyTuple> {
        self.nodes.clone_ref(py)
    }

    #[getter]
    fn root(&self, py: Python<'_>) -> Py<PyAny> {
        self.root.clone_ref(py)
    }

    /// Return the node with an id.
    fn node<'py>(&self, py: Python<'py>, node_id: &str) -> PyResult<Bound<'py, PyAny>> {
        match self
            .graph
            .nodes
            .iter()
            .position(|node| node.id() == node_id)
        {
            Some(position) => self.nodes.bind(py).get_item(position),
            None => Err(PyKeyError::new_err(node_id.to_string())),
        }
    }

    /// Validate ids, ports, types, columns, cycles, the root, and, when
    /// given, that every node has a registered runtime.
    #[pyo3(signature = (runtime_keys=None))]
    fn validate(&self, runtime_keys: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        topology::validate_graph(&self.graph.nodes, &self.graph.root)
            .map_err(GraphValidationError::new_err)?;
        if let Some(keys) = runtime_keys {
            let mut missing: Vec<&str> = Vec::new();
            for node in &self.graph.nodes {
                if !keys.contains(node.type_name())? && !missing.contains(&node.type_name()) {
                    missing.push(node.type_name());
                }
            }
            if !missing.is_empty() {
                missing.sort();
                return Err(GraphValidationError::new_err(format!(
                    "physical graph needs unregistered runtimes {missing:?}"
                )));
            }
        }
        Ok(())
    }

    /// Check that every model node uses the selected backend.
    fn validate_backend(&self, backend: &str) -> PyResult<()> {
        let mut wrong: Vec<String> = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                node.backend()
                    .filter(|found| *found != backend)
                    .map(|found| format!("{}:{found}", node.id()))
            })
            .collect();
        wrong.sort();
        if wrong.is_empty() {
            return Ok(());
        }
        Err(GraphValidationError::new_err(format!(
            "physical graph selects backend {backend:?} but has model nodes {wrong:?}"
        )))
    }

    /// Return the nodes, each after all its inputs.
    fn topological_nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.validate(None)?;
        self.by_position(py, &topology::topological_nodes(&self.graph.nodes))
    }

    /// Return the nodes with one type name.
    fn nodes_by_type<'py>(
        &self,
        py: Python<'py>,
        type_name: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let positions: Vec<usize> = (0..self.graph.nodes.len())
            .filter(|position| self.graph.nodes[*position].type_name() == type_name)
            .collect();
        self.by_position(py, &positions)
    }

    /// Group compatible per-document operators into pipelines.
    ///
    /// A pipeline runs the same way on one GPU and on each worker of
    /// several, over that worker's share of the documents.
    ///
    /// Args:
    ///     completed: Nodes whose outputs are already available.
    ///
    /// Returns:
    ///     A mapping from each node id in a pipeline to the pipeline's
    ///     nodes, in order.
    #[pyo3(signature = (completed=Vec::new()))]
    fn pipelines<'py>(
        &self,
        py: Python<'py>,
        completed: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.validate(None)?;
        let pipelines = topology::build_pipelines(&self.graph.nodes, &completed);
        let found = PyDict::new(py);
        let groups = pipelines
            .members
            .iter()
            .map(|members| self.by_position(py, members))
            .collect::<PyResult<Vec<_>>>()?;
        for (node_id, index) in &pipelines.of {
            found.set_item(node_id, &groups[*index])?;
        }
        Ok(found)
    }

    /// Return the ids of the model nodes whose documents' KV is already
    /// resident: each pipeline member after the first, and a pipeline's
    /// first node on a table whose previous pipeline kept its KV under
    /// the same document head.
    #[pyo3(signature = (completed=Vec::new()))]
    fn resident_inputs<'py>(
        &self,
        py: Python<'py>,
        completed: Vec<String>,
    ) -> PyResult<Bound<'py, PyFrozenSet>> {
        self.validate(None)?;
        let pipelines = topology::build_pipelines(&self.graph.nodes, &completed);
        PyFrozenSet::new(py, topology::resident_inputs(&self.graph.nodes, &pipelines))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PhysicalGraph>()
            .is_ok_and(|other| other.get().graph == self.graph)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "PhysicalGraph(nodes={}, root={})",
            self.nodes.bind(py).repr()?,
            self.root.bind(py).repr()?
        ))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let decode = py
            .import("_quail_native")?
            .getattr("PhysicalGraph")?
            .getattr("from_json")?;
        PyTuple::new(py, [decode, PyTuple::new(py, [self.to_json()])?.into_any()])
    }
}

/// Read a value of a physical plan from JSON: a node, a spec, a port, or
/// an expression as its class, and any other value as Python's.
#[pyfunction]
pub fn decode_plan_value<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if let Some(tag) = value.get("type").and_then(Value::as_str)
        && value.get("outputs").is_some()
        && !is_known_tag(tag)
    {
        let node: physical::PhysicalNode = serde_json::from_value(value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        return node_object(py, Arc::new(node));
    }
    to_python(py, &value)
}

/// Write a value of a physical plan as JSON: the inverse of
/// `decode_plan_value`.
#[pyfunction]
pub fn encode_plan_value(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(from_python(value)?.to_string())
}

/// Typed input ports that read the given outputs, named `input:<index>`,
/// each carrying the value type its source port's name implies.
#[pyfunction]
pub fn input_ports<'py>(
    py: Python<'py>,
    sources: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let sources = sources
        .try_iter()?
        .map(|source| Tagged::held_value::<physical::PortRef>(&source?))
        .collect::<PyResult<Vec<_>>>()?;
    let ports = topology::input_ports(&sources)
        .iter()
        .map(|port| to_python(py, &serde_json::to_value(port).expect("a port writes")))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, ports)
}

/// Register the graph class, its error, and the plan value functions.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PhysicalGraph>()?;
    module.add(
        "GraphValidationError",
        py.get_type::<GraphValidationError>(),
    )?;
    module.add_function(wrap_pyfunction!(decode_plan_value, module)?)?;
    module.add_function(wrap_pyfunction!(encode_plan_value, module)?)?;
    module.add_function(wrap_pyfunction!(input_ports, module)?)?;
    Ok(())
}
