"""Integration tests for MCP subprocess connection (H9).

Starts a minimal MCP stub server as a subprocess (stdio), connects via
MCPClientManager, retrieves tools, and calls query_knowledge_hub once.
Asserts that the result is parseable (RAGToolWrapper returns a non-empty
list of contexts). Does not require a real RAG server.

Design reference: DEV_SPEC H9.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.config import MCPConfig, MCPConnectionConfig
from src.tools.mcp_client import MCPClientManager


def _stub_script_path() -> Path:
    """Path to the MCP stub server script (tests/fixtures/mcp_stub_server.py)."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "mcp_stub_server.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.mark.slow
@pytest.mark.asyncio
async def test_mcp_subprocess_connect_and_query_knowledge_hub() -> None:
    """Start stub MCP server subprocess, get tools, call query_knowledge_hub; assert result is parseable."""
    stub_path = _stub_script_path()
    repo_root = _repo_root()
    assert stub_path.exists(), f"MCP stub script not found: {stub_path}"

    conn = MCPConnectionConfig(
        transport="stdio",
        command=sys.executable,
        args=[str(stub_path)],
        cwd=str(repo_root),
        mcp_stderr="devnull",
    )
    mcp_config = MCPConfig(connections={"rag_server": conn})

    manager = MCPClientManager(mcp_config)
    tools = await manager.get_tools(server_name="rag_server")
    assert tools, "get_tools should return at least one tool"

    query_tool = next((t for t in tools if t.name == "query_knowledge_hub"), None)
    assert query_tool is not None, "query_knowledge_hub tool should be present"

    raw = await query_tool.ainvoke({"query": "test query", "top_k": 3})
    assert raw is not None, "tool ainvoke should return a value"
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if "results" in text or "stub" in text.lower():
                    break
        else:
            assert False, "raw result should be parseable (content with results or stub text)"
    elif isinstance(raw, dict):
        assert "results" in raw or "content" in raw or "citations" in raw, "result should be parseable"
    else:
        assert isinstance(raw, str) and len(raw) > 0, "result should be non-empty string or structured"
