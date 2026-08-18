"""AI SQL entry point: Snowflake AISQL syntax, filters and joins only."""

from .compile import compile_sql

__all__ = ["compile_sql"]
