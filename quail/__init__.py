"""Quail: a declarative query engine for AI_FILTER and AI_JOIN."""

from quail.builder import col, prompt
from quail.catalog import DocumentProvider, ScanRequest, TableProvider
from quail.extensions import ExtensionRegistry
from quail.physical import NodeCodec, PhysicalGraph, PhysicalNode
from quail.planner.plan import EngineConfig
from quail.runtime.compute import (
    ComputeProvider,
    InProcessComputeProvider,
    ModalComputeProvider,
    QueryRequest,
)
from quail.runtime.result import QueryResult
from quail.runtime.session import Query, RefusalError, Session
from quail.sqlfront import SQLDialect

__all__ = [
    "ComputeProvider",
    "DocumentProvider",
    "EngineConfig",
    "ExtensionRegistry",
    "InProcessComputeProvider",
    "ModalComputeProvider",
    "NodeCodec",
    "PhysicalGraph",
    "PhysicalNode",
    "Query",
    "QueryRequest",
    "QueryResult",
    "RefusalError",
    "ScanRequest",
    "Session",
    "SQLDialect",
    "TableProvider",
    "col",
    "prompt",
]
