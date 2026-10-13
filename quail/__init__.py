"""Quail: a declarative query engine for AI_FILTER and AI_JOIN."""

from quail.api import Query, RefusalError, Session
from quail.catalog import DocumentProvider, ScanRequest, TableProvider
from quail.execution.result import QueryResult
from quail.extensions import ExtensionRegistry
from quail.frontend.expressions import col, prompt
from quail.frontend.sql import SQLDialect
from quail.logical import (
    SHARED_PRE,
    ColumnRef,
    bind_join_prompt,
    bind_prompt,
    render_join_prompt_text,
)
from quail.physical import NodeCodec, PhysicalGraph, PhysicalNode
from quail.planner.estimate import SpeedOfLightEstimate, speed_of_light_estimate
from quail.planner.plan import EngineConfig
from quail.planner.prompt_lowering import lower_prompt, true_false_ids
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
    "SpeedOfLightEstimate",
    "SQLDialect",
    "TableProvider",
    "bind_join_prompt",
    "bind_prompt",
    "col",
    "lower_prompt",
    "prompt",
    "render_join_prompt_text",
    "speed_of_light_estimate",
    "true_false_ids",
]
