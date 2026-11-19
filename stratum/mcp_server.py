from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from typing import Any

from .schemas import ALL_TOOLS
from .tools.tool_audit import run_audit
from .tools.tool_compare import run_compare
from .tools.tool_contamination import run_contamination
from .tools.tool_decisions import (
    run_create_decision,
    run_decisions,
    run_journal,
    run_log_event,
    run_next_action,
    run_resolve_decision,
)
from .tools.tool_drift import run_drift
from .tools.tool_drift_current import run_drift_current
from .tools.tool_engineer import run_engineer
from .tools.tool_explain import run_explain
from .tools.tool_export import run_export
from .tools.tool_join_check import run_join_check
from .tools.tool_label_quality import run_label_quality
from .tools.tool_leakage_deep import run_leakage_deep
from .tools.tool_predict import run_predict
from .tools.tool_profile import run_profile
from .tools.tool_report import run_report
from .tools.tool_schema_check import run_schema_check
from .tools.tool_segments import run_segments
from .tools.tool_group_evaluation import run_group_evaluation
from .tools.tool_split import run_split
from .tools.tool_stability import run_stability
from .tools.tool_status import run_status
from .tools.tool_train import run_train
from .tools.tool_validate import run_validate


HANDLERS = {
    "stratum_profile": run_profile,
    "stratum_schema_check": run_schema_check,
    "stratum_split": run_split,
    "stratum_drift": run_drift,
    "stratum_drift_current": run_drift_current,
    "stratum_audit": run_audit,
    "stratum_leakage_deep": run_leakage_deep,
    "stratum_label_quality": run_label_quality,
    "stratum_contamination": run_contamination,
    "stratum_join_check": run_join_check,
    "stratum_create_decision": run_create_decision,
    "stratum_decisions": run_decisions,
    "stratum_resolve_decision": run_resolve_decision,
    "stratum_log_event": run_log_event,
    "stratum_journal": run_journal,
    "stratum_next_action": run_next_action,
    "stratum_engineer": run_engineer,
    "stratum_train": run_train,
    "stratum_validate": run_validate,
    "stratum_segments": run_segments,
    "stratum_group_evaluation": run_group_evaluation,
    "stratum_stability": run_stability,
    "stratum_predict": run_predict,
    "stratum_explain": run_explain,
    "stratum_compare": run_compare,
    "stratum_export": run_export,
    "stratum_status": run_status,
    "stratum_report": run_report,
}


async def dispatch_tool(name: str, arguments: dict[str, Any] | None = None) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = handler(**(arguments or {}))
        if inspect.isawaitable(result):
            result = await result
        return result
    except FileNotFoundError as exc:
        return {"error": f"File not found: {exc}"}
    except (TypeError, ValueError) as exc:
        return {"error": f"Invalid input: {exc}"}
    except Exception as exc:
        return {"error": f"Internal error: {type(exc).__name__}: {exc}"}


def create_server():
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("stratum-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**schema) for schema in ALL_TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = await dispatch_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


async def _run() -> None:
    from mcp.server.stdio import stdio_server

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stratum-mcp",
        description="Run the Stratum MCP server over stdio.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the Stratum package version and exit.",
    )
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return
    asyncio.run(_run())


if __name__ == "__main__":
    main(sys.argv[1:])
