"""Build a physical dependency graph for a statement's query stages."""

from quail.logical import cte_schema
from quail.physical import InputPort, PortRef, QueryStage, ValueType
from quail.physical.plan import PhysicalPlan


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
