"""Unit tests for MCP Client (B1).

Verifies that ``MCPClientManager`` correctly builds connection dicts from
config, delegates to ``MultiServerMCPClient.get_tools()``, and handles
error conditions — all with mocked stdio/http transports (no real server).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import MCPConfig, MCPConnectionConfig
from src.tools.mcp_client import (
    MCPClientManager,
    _build_connection_dict,
    build_connections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stdio_conn_config() -> MCPConnectionConfig:
    """A valid stdio connection configuration."""
    return MCPConnectionConfig(
        transport="stdio",
        command="python",
        args=["-m", "rag_server"],
    )


@pytest.fixture()
def http_conn_config() -> MCPConnectionConfig:
    """A valid streamable-http connection configuration."""
    return MCPConnectionConfig(
        transport="streamable-http",
        url="http://localhost:8000/mcp",
    )


@pytest.fixture()
def mcp_config_stdio(stdio_conn_config: MCPConnectionConfig) -> MCPConfig:
    """MCPConfig with a single stdio server."""
    return MCPConfig(connections={"rag_server": stdio_conn_config})


@pytest.fixture()
def mcp_config_multi(
    stdio_conn_config: MCPConnectionConfig,
    http_conn_config: MCPConnectionConfig,
) -> MCPConfig:
    """MCPConfig with both stdio and http servers."""
    return MCPConfig(
        connections={
            "rag_server": stdio_conn_config,
            "remote_rag": http_conn_config,
        }
    )


def _make_fake_tool(name: str) -> MagicMock:
    """Create a mock LangChain BaseTool with a given name."""
    tool = MagicMock()
    tool.name = name
    return tool


# ---------------------------------------------------------------------------
# _build_connection_dict
# ---------------------------------------------------------------------------


class TestBuildConnectionDict:
    """Tests for the low-level connection dict builder."""

    def test_stdio_connection_dict(
        self, stdio_conn_config: MCPConnectionConfig
    ) -> None:
        """Stdio config should produce a dict with transport/command/args."""
        result = _build_connection_dict(stdio_conn_config)
        assert result == {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "rag_server"],
        }

    def test_streamable_http_connection_dict(
        self, http_conn_config: MCPConnectionConfig
    ) -> None:
        """Streamable-http config should produce a dict with transport/url."""
        result = _build_connection_dict(http_conn_config)
        assert result == {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        }

    def test_unsupported_transport_raises_value_error(self) -> None:
        """An unknown transport should raise ValueError."""
        conn = MCPConnectionConfig.model_construct(
            transport="grpc",
            command="python",
            args=[],
            url=None,
        )
        with pytest.raises(ValueError, match="Unsupported MCP transport"):
            _build_connection_dict(conn)

    def test_streamable_http_underscore_alias(self) -> None:
        """Transport 'streamable_http' (underscore) should also work."""
        conn = MCPConnectionConfig.model_construct(
            transport="streamable_http",
            command=None,
            args=[],
            url="http://localhost:8000/mcp",
        )
        result = _build_connection_dict(conn)
        assert result["transport"] == "streamable_http"
        assert result["url"] == "http://localhost:8000/mcp"


# ---------------------------------------------------------------------------
# build_connections
# ---------------------------------------------------------------------------


class TestBuildConnections:
    """Tests for the full connections dict builder."""

    def test_single_stdio_server(
        self, mcp_config_stdio: MCPConfig
    ) -> None:
        """Single stdio server should produce one-entry dict."""
        result = build_connections(mcp_config_stdio)
        assert len(result) == 1
        assert "rag_server" in result
        assert result["rag_server"]["transport"] == "stdio"

    def test_multi_server(self, mcp_config_multi: MCPConfig) -> None:
        """Multiple servers should all appear in the result."""
        result = build_connections(mcp_config_multi)
        assert len(result) == 2
        assert "rag_server" in result
        assert "remote_rag" in result
        assert result["rag_server"]["transport"] == "stdio"
        assert result["remote_rag"]["transport"] == "streamable_http"

    def test_empty_connections(self) -> None:
        """Empty connections dict should produce empty result."""
        config = MCPConfig(connections={})
        result = build_connections(config)
        assert result == {}


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------


class TestMCPClientManager:
    """Tests for the main MCPClientManager class."""

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    def test_init_creates_client(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """Manager should instantiate MultiServerMCPClient with correct connections."""
        manager = MCPClientManager(mcp_config_stdio)

        mock_client_cls.assert_called_once()
        call_args = mock_client_cls.call_args
        connections = call_args[0][0]
        assert "rag_server" in connections
        assert connections["rag_server"]["transport"] == "stdio"
        assert manager.config is mcp_config_stdio

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_get_tools_returns_tools(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """get_tools() should delegate to client.get_tools() and return tools."""
        fake_tools = [
            _make_fake_tool("query_knowledge_hub"),
            _make_fake_tool("list_collections"),
            _make_fake_tool("get_document_summary"),
        ]
        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(return_value=fake_tools)
        mock_client_cls.return_value = mock_instance

        manager = MCPClientManager(mcp_config_stdio)
        tools = await manager.get_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        }
        mock_instance.get_tools.assert_awaited_once_with(server_name=None)

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_get_tools_with_server_name(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """get_tools(server_name=...) should forward the server_name."""
        fake_tools = [_make_fake_tool("query_knowledge_hub")]
        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(return_value=fake_tools)
        mock_client_cls.return_value = mock_instance

        manager = MCPClientManager(mcp_config_stdio)
        tools = await manager.get_tools(server_name="rag_server")

        assert len(tools) == 1
        mock_instance.get_tools.assert_awaited_once_with(
            server_name="rag_server"
        )

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_get_tools_unknown_server_raises_value_error(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """get_tools() with an unknown server name should raise ValueError."""
        mock_client_cls.return_value = MagicMock()

        manager = MCPClientManager(mcp_config_stdio)

        with pytest.raises(ValueError, match="Unknown MCP server"):
            await manager.get_tools(server_name="nonexistent")

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_get_tools_connection_error(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """get_tools() should wrap underlying errors as ConnectionError."""
        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(
            side_effect=RuntimeError("server unreachable")
        )
        mock_client_cls.return_value = mock_instance

        manager = MCPClientManager(mcp_config_stdio)

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await manager.get_tools()

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    def test_get_server_names(
        self,
        mock_client_cls: MagicMock,
        mcp_config_multi: MCPConfig,
    ) -> None:
        """get_server_names() should return sorted names."""
        mock_client_cls.return_value = MagicMock()

        manager = MCPClientManager(mcp_config_multi)
        names = manager.get_server_names()

        assert names == ["rag_server", "remote_rag"]

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_get_tools_empty_result(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """get_tools() should handle an empty tool list gracefully."""
        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(return_value=[])
        mock_client_cls.return_value = mock_instance

        manager = MCPClientManager(mcp_config_stdio)
        tools = await manager.get_tools()

        assert tools == []

    @patch("src.tools.mcp_client.MultiServerMCPClient")
    async def test_tools_are_invocable(
        self,
        mock_client_cls: MagicMock,
        mcp_config_stdio: MCPConfig,
    ) -> None:
        """Returned tools should support ainvoke (mock demonstration)."""
        fake_tool = _make_fake_tool("query_knowledge_hub")
        fake_tool.ainvoke = AsyncMock(return_value="mocked result")

        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(return_value=[fake_tool])
        mock_client_cls.return_value = mock_instance

        manager = MCPClientManager(mcp_config_stdio)
        tools = await manager.get_tools()

        result = await tools[0].ainvoke({"query": "What is LoRA?"})
        assert result == "mocked result"
        fake_tool.ainvoke.assert_awaited_once_with({"query": "What is LoRA?"})
