"""Quail: a declarative query engine for AI_FILTER and AI_JOIN."""

from quail.api import Query, RefusalError, Session, SessionContext
from quail.api.compile import SQLDialect
from quail.api.config import EngineConfig
from quail.api.prompts import bind_join_prompt, bind_prompt
from quail.api.sol import SpeedOfLightEstimate, speed_of_light_estimate
from quail.catalog import DocumentProvider, ScanRequest, TableProvider
from quail.common.expressions import ColumnRef
from quail.common.prompts import SHARED_PRE, true_false_ids
from quail.execution.result import QueryResult
from quail.extensions import ExtensionRegistry
from quail.physical import NodeCodec, PhysicalGraph, PhysicalNode
from quail.physical.lowering import lower_prompt
from quail.remote.client import QueryRun
from quail.remote.records import QueryFailedError, QueryStatus

__all__ = [
    "SHARED_PRE",
    "ColumnRef",
    "DocumentProvider",
    "EngineConfig",
    "ExtensionRegistry",
    "NodeCodec",
    "PhysicalGraph",
    "PhysicalNode",
    "Query",
    "QueryFailedError",
    "QueryResult",
    "QueryRun",
    "QueryStatus",
    "RefusalError",
    "ScanRequest",
    "Session",
    "SessionContext",
    "SpeedOfLightEstimate",
    "SQLDialect",
    "TableProvider",
    "bind_join_prompt",
    "bind_prompt",
    "lower_prompt",
    "speed_of_light_estimate",
    "true_false_ids",
]
