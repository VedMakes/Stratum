import asyncio
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
