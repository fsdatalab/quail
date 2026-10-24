"""Build a physical dependency graph for a statement's query stages."""

from quail.logical import column_name, cte_schema
from quail.physical import InputPort, PortRef, QueryStage, Scan, ValueType
from quail.planner.graph.results import result_nodes
from quail.planner.plan import PhysicalPlan


def stage_plan(logical, plans, assumptions=None) -> PhysicalPlan:
    """Connect planned stages to the intermediate relations they read."""
    definitions = dict(logical.stages())
    nodes = []
    for name, plan in plans.items():
        dependencies = dict.fromkeys(
            scan.provider for scan in definitions[name].operators().scans
            if scan.provider in definitions and scan.provider)
        nodes.append(QueryStage(
            node_id=f"stage:{name}", name=name,
            inputs=tuple(InputPort(provider, ValueType.ROWS,
                                   PortRef(f"stage:{provider}", "rows"))
                         for provider in dependencies),
            graph=plan.graph, settings=dict(plan.settings),
            estimates=dict(plan.estimates),
            estimated_seconds=plan.estimated_seconds,
            columns=tuple((cte_schema(definitions[name]) if name
                           else definitions[name].schema()).names),
            assumptions=tuple(assumptions.get(name, ())) if assumptions else (),
        ))
    return PhysicalPlan(
        model=plan.model, device=plan.device, gpus=plan.gpus, backend=plan.backend,
        nodes=tuple(nodes), root=PortRef("stage:", "rows"),
        estimated_seconds=sum(node.estimated_seconds for node in nodes),
        estimates={node.node_id: {"seconds": node.estimated_seconds}
                   for node in nodes}, logical_plan=logical)


def relational_plan(logical, config, num_documents) -> PhysicalPlan:
    """Plan a relational stage using the ordinary scan and result operators."""
    (scan,) = logical.operators().scans
    source = Scan(node_id=f"scan:{scan.alias}", alias=scan.alias,
                  input_id=scan.alias, num_documents=num_documents,
                  predicates=scan.predicates)
    nodes = (source, *result_nodes(
        logical.result, (PortRef(source.node_id, f"rows:{scan.alias}"),),
        tuple(column_name(column) for column in logical.projection.columns),
        logical.projection.packing()))
    return PhysicalPlan(
        model=config.model, device=config.device, gpus=config.gpus,
        backend=config.backend, nodes=nodes, logical_plan=logical)
