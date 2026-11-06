"""Model region inputs: relations DataFusion computes for a region.

A CTE or FROM subquery that groups or windows rows runs in DataFusion.
The model region reads its rows as a table of the same name, which this
module adds to the region's catalog: the rows DataFusion produced when
the region runs, or an empty table of the input's schema for planning
before a run.
"""

import weakref
from contextlib import closing

import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog import Catalog, MemoryTableProvider, ScanRequest
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


def source_rows(catalog, name: str, table) -> pa.Array | None:
    """Each row's position in the registered table an input filters.

    DataFusion names an input that holds some of a registered table's
    rows ``<table>__<alias>``; its rows are found in the table by the
    table's id column. Returns None for any other input.
    """
    base = name.split("__", 1)[0]
    if table is None or "__" not in name or base not in catalog:
        return None
    provider = catalog.get(base)
    id_col = getattr(provider, "id_col", "")
    if not id_col or id_col not in table.column_names:
        return None
    with closing(provider.scan(ScanRequest(columns=(id_col,)))) as reader:
        ids = reader.read_all().column(id_col)
    return pc.index_in(table.column(id_col), value_set=ids).cast(pa.int32())


def region_query(session, logical: LogicalPlan, *, tables: dict | None = None,
                 **options):
    """Return the Query for a model region, over its input tables.

    Args:
        session: The session whose tables and model the region uses.
        logical: The region's logical plan.
        tables: Each input's rows as a pyarrow Table, by name, or None
            to plan before the inputs exist.
        **options: Passed to Query.

    Returns:
        A Query whose catalog holds the session's tables and the inputs.
    """
    from quail.api.query import Query

    scans = input_scans(logical, session.catalog)
    if not scans:
        return Query(session, logical, **options)
    providers = dict(session.catalog.providers)
    assumed = {}
    rows = {}
    for name, scan in scans.items():
        if tables is not None and name in tables:
            table = tables[name]
            found = source_rows(session.catalog, name, table)
            if found is not None:
                # DataFusion scans in parallel; the region reads the rows in
                # the table's order
                order = pc.sort_indices(found)
                table, rows[name] = table.take(order), pc.take(found, order)
        else:
            table = scan.schema.empty_table()
            assumed[name] = lambda _column: TokenLengths(
                [ASSUMED_INPUT_TOKENS] * ASSUMED_INPUT_ROWS)
        providers[name] = MemoryTableProvider(table, id_col="",
                                              identity=f"input:{name}")
    catalog = Catalog(providers)
    store = session.inputs.over(catalog, assumed)
    query = Query(session, logical, catalog=catalog, inputs=store, **options)
    query.source_rows = rows
    # the store's token files live as long as the query
    weakref.finalize(query, store.close)
    return query
