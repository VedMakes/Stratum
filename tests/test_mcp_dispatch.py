import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from stratum.mcp_server import dispatch_tool
from stratum.schemas import ALL_TOOLS


def test_unknown_tool_is_structured_error():
    result = asyncio.run(dispatch_tool("missing", {}))
    assert result == {"error": "Unknown tool: missing"}


def test_all_tools_have_handlers():
    from stratum.mcp_server import HANDLERS

    assert {tool["name"] for tool in ALL_TOOLS} == set(HANDLERS)


def test_cli_help_and_version_exit_without_server(capsys):
    from stratum.mcp_server import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "Run the Stratum MCP server" in capsys.readouterr().out

    main(["--version"])
    assert capsys.readouterr().out.strip()


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP extra is not installed")
def test_stdio_subprocess_initializes_and_lists_tools():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise():
        command = Path(sys.executable).with_name("stratum-mcp")
        if os.name == "nt":
            command = command.with_suffix(".exe")
        assert command.is_file(), f"Installed MCP entrypoint not found: {command}"
        params = StdioServerParameters(
            command=str(command),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=30)
                tools = await asyncio.wait_for(session.list_tools(), timeout=30)
                assert {tool.name for tool in tools.tools} == {
                    tool["name"] for tool in ALL_TOOLS
                }

    asyncio.run(exercise())
