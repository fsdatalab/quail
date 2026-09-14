"""Quail: a declarative query engine for AI_FILTER and AI_JOIN."""

from quail.catalog import DocumentProvider, ScanRequest, TableProvider
from quail.execution.result import QueryResult
from quail.execution.session import Query, RefusalError, Session
from quail.extensions import ExtensionRegistry
from quail.frontend.builder import col, prompt
from quail.frontend.sql import SQLDialect
from quail.logical import (
    SHARED_PRE,
    ColumnRef,
    bind_join_prompt,
    bind_prompt,
    render_join_prompt_text,
    true_false_ids,
)
from quail.physical import NodeCodec, PhysicalGraph, PhysicalNode
from quail.planner.estimate import SpeedOfLightEstimate, speed_of_light_estimate
from quail.planner.plan import EngineConfig

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
    "QueryResult",
    "RefusalError",
    "ScanRequest",
    "Session",
    "SpeedOfLightEstimate",
    "SQLDialect",
    "TableProvider",
    "bind_join_prompt",
    "bind_prompt",
    "col",
    "prompt",
    "render_join_prompt_text",
    "speed_of_light_estimate",
    "true_false_ids",
]
