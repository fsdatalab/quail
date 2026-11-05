"""Model region inputs: relations DataFusion computes for a region.

A CTE or FROM subquery that groups or windows rows runs in DataFusion.
The model region reads its rows as a table of the same name, which this
module adds to the region's catalog: the rows DataFusion produced when
the region runs, or an empty table of the input's schema for planning
before a run.
"""

import weakref

from quail.catalog import Catalog, MemoryTableProvider
from quail.common.tokens import TokenLengths
from quail.logical import LogicalPlan, Scan

# planning before a run assumes this many rows of this many tokens each
ASSUMED_INPUT_ROWS = 1_000
ASSUMED_INPUT_TOKENS = 32


def input_scans(logical: LogicalPlan, catalog) -> dict:
    """Return each Scan of a region input, by its table name.

    A region input is a table the catalog does not hold and no CTE
    stage defines.
    """
    stages = {name for name, _ in logical.stages() if name}
    return {node.provider: node for node in logical.walk()
            if isinstance(node, Scan) and node.provider not in catalog
            and node.provider not in stages}


def region_query(session, logical: LogicalPlan, *, tables: dict | None = None,
                 **options):
    """Return the Query for a model region, over its input tables.

    Args:
        session: The session whose tables and model the region uses.
        logical: The region's logical plan.
        tables: Each input's rows as Arrow IPC stream bytes, by name, or
            None to plan before the inputs exist.
        **options: Passed to Query.

    Returns:
        A Query whose catalog holds the session's tables and the inputs.
    """
    from quail.api.context import ipc_table
    from quail.api.query import Query

    scans = input_scans(logical, session.catalog)
    if not scans:
        return Query(session, logical, **options)
    providers = dict(session.catalog.providers)
    assumed = {}
    for name, scan in scans.items():
        if tables is not None and name in tables:
            table = ipc_table(tables[name])
        else:
            table = scan.schema.empty_table()
            assumed[name] = lambda _column: TokenLengths(
                [ASSUMED_INPUT_TOKENS] * ASSUMED_INPUT_ROWS)
        providers[name] = MemoryTableProvider(table, id_col="",
                                              identity=f"input:{name}")
    catalog = Catalog(providers)
    store = session.inputs.over(catalog, assumed)
    query = Query(session, logical, catalog=catalog, inputs=store, **options)
    # the store's token files live as long as the query
    weakref.finalize(query, store.close)
    return query
