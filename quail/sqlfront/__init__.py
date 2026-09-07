"""AI SQL entry point for supported Snowflake and BigQuery syntax."""

from .compile import SQLDialect, compile_sql

__all__ = ["SQLDialect", "compile_sql"]
