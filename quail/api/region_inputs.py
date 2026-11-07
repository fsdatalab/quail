"""Model region inputs: relations DataFusion computes for a region.

A CTE or FROM subquery that groups or windows rows runs in DataFusion.
The model region reads its rows as a table of the same name, which this
module adds to the region's catalog: the rows DataFusion produced when
the region runs, or an empty table of the input's schema for planning
before a run.
"""

import base64
import json
from contextlib import closing

import _quail_native
import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog import (
    INPUT_SEPARATOR,
    Catalog,
    MemoryTableProvider,
    ScanRequest,
    table_name,
)

# planning before a run assumes this many rows of this many tokens each
ASSUMED_INPUT_ROWS = 1_000
ASSUMED_INPUT_TOKENS = 32


def input_scans(facts: dict, catalog) -> dict:
    """Return each scan of a region input, by its table name.

    A region input is a table the catalog does not hold.

    Args:
        facts: The region's facts, as ``region_facts`` returns them.
        catalog: The session's catalog.
    """
    return {scan["provider"]: scan for scan in facts["scans"]
            if scan["provider"] not in catalog}


def scan_schema(scan: dict) -> pa.Schema:
    """Return the Arrow schema a scan's facts carry."""
    return pa.ipc.read_schema(pa.py_buffer(base64.b64decode(scan["schema_ipc"])))


def source_rows(catalog, name: str, table) -> pa.Array | None:
    """Each row's position in the registered table an input filters.

    DataFusion names an input that holds some of a registered table's
    rows ``<table>__<alias>``; its rows are found in the table by the
    table's id column. Returns None for any other input.
    """
    base = table_name(name)
    if table is None or INPUT_SEPARATOR not in name or base not in catalog:
        return None
    provider = catalog.get(base)
    id_col = getattr(provider, "id_col", "")
    if not id_col or id_col not in table.column_names:
        return None
    with closing(provider.scan(ScanRequest(columns=(id_col,)))) as reader:
        ids = reader.read_all().column(id_col)
    return pc.index_in(table.column(id_col), value_set=ids).cast(pa.int32())


def region_query(session, operators: str, *, tables: dict | None = None,
                 keep_order: bool = False, **options):
    """Return the RegionQuery for a model region, over its input tables.

    Args:
        session: The session whose tables and model the region uses.
        operators: The region's operators, as Rust's JSON.
        tables: Each input's rows as a pyarrow Table, by name, or None
            to plan before the inputs exist.
        keep_order: Read each input in the order given, as a plan Rust
            made over these rows expects, rather than in the table's order.
        **options: Passed to RegionQuery.

    Returns:
        A RegionQuery whose catalog holds the session's tables and the inputs.
    """
    from quail.api.query import RegionQuery

    scans = input_scans(json.loads(_quail_native.region_facts(operators)),
                        session.catalog)
    if not scans:
        return RegionQuery(session, operators, **options)
    providers = dict(session.catalog.providers)
    assumed = {}
    rows, identities = {}, {}
    for name, scan in scans.items():
        if tables is not None and name in tables:
            table = tables[name]
            found = source_rows(session.catalog, name, table)
            if found is not None and keep_order:
                rows[name] = found
            elif found is not None:
                # DataFusion scans in parallel; the region reads the rows in
                # the table's order
                order = pc.sort_indices(found)
                table, rows[name] = table.take(order), pc.take(found, order)
            if found is not None:
                base = session.catalog.get(table_name(name))
                if len(found) == base.statistics().num_rows:
                    # the whole table: its token store is kept for later queries
                    identities[name] = base.content_identity()
        else:
            table = scan_schema(scan).empty_table()
            assumed[name] = ASSUMED_INPUT_ROWS
        providers[name] = MemoryTableProvider(table, id_col="",
                                              identity=f"input:{name}")
    catalog = Catalog(providers)
    query = RegionQuery(session, operators, catalog=catalog, assumed_rows=assumed,
                        **options)
    query.source_rows = rows
    query.input_identities = identities
    return query
