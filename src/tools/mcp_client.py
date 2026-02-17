"""MCP Client: connect to RAG Server via stdio or streamable-http.

Wraps ``langchain-mcp-adapters``'s ``MultiServerMCPClient`` and provides a
config-driven ``MCPClientManager`` that reads ``MCPConfig`` from settings,
builds connection dicts, and exposes ``get_tools()`` to retrieve
LangChain-compatible tools for the agent graph.

Design reference: DEV_SPEC §3.2, §5.3 (tools/mcp_client.py), §7.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import MCPConfig, MCPConnectionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transport name mapping (config YAML → langchain-mcp-adapters)
# ---------------------------------------------------------------------------

_TRANSPORT_MAP: dict[str, str] = {
    "stdio": "stdio",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
}
"""Map user-facing transport names to the literal values expected by
``langchain-mcp-adapters`` connection TypedDicts."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_connection_dict(conn: MCPConnectionConfig) -> dict[str, Any]:
    """Convert a single ``MCPConnectionConfig`` to the dict format expected
    by ``MultiServerMCPClient``.

    Args:
        conn: Validated MCP connection configuration.

    Returns:
        A dict suitable as a value in the ``connections`` parameter of
        ``MultiServerMCPClient``.

    Raises:
        ValueError: If the transport type is unsupported.
    """
    transport = _TRANSPORT_MAP.get(conn.transport)
    if transport is None:
        raise ValueError(
            f"Unsupported MCP transport: {conn.transport!r}. "
            f"Supported: {sorted(_TRANSPORT_MAP.keys())}"
        )

    if transport == "stdio":
        out: dict[str, Any] = {
            "transport": transport,
            "command": conn.command,
            "args": conn.args,
        }
        if conn.cwd:
            out["cwd"] = conn.cwd
        if conn.env:
            # Merge with current env so RAG subprocess still has PATH etc.
            env = os.environ.copy()
            env.update(conn.env)
            out["env"] = env
        return out

    # streamable_http
    return {
        "transport": transport,
        "url": conn.url,
    }


def build_connections(mcp_config: MCPConfig) -> dict[str, dict[str, Any]]:
    """Build the full connections dict from ``MCPConfig``.

    Args:
        mcp_config: Validated MCP configuration with named connections.

    Returns:
        A dict mapping server names to connection configuration dicts
        ready for ``MultiServerMCPClient``.
    """
    connections: dict[str, dict[str, Any]] = {}
    for name, conn in mcp_config.connections.items():
        connections[name] = _build_connection_dict(conn)
        logger.debug("Prepared MCP connection %r (%s)", name, conn.transport)
    return connections


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------


class MCPClientManager:
    """Config-driven manager for MCP server connections.

    Reads ``MCPConfig`` (from ``load_settings().mcp``), constructs the
    underlying ``MultiServerMCPClient``, and exposes async methods for
    retrieving tools.

    Attributes:
        config: The MCP configuration used to build connections.
        client: The underlying ``MultiServerMCPClient`` instance.

    Example::

        from src.config import load_settings

        settings = load_settings()
        manager = MCPClientManager(settings.mcp)
        tools = await manager.get_tools()
    """

    def __init__(self, mcp_config: MCPConfig) -> None:
        """Initialize the manager and construct the underlying client.

        Args:
            mcp_config: Validated ``MCPConfig`` from application settings.
        """
        self.config = mcp_config
        self._connections = build_connections(mcp_config)
        self.client = MultiServerMCPClient(self._connections)
        logger.info(
            "MCPClientManager initialized with %d server(s): %s",
            len(self._connections),
            ", ".join(self._connections.keys()),
        )

    async def get_tools(
        self,
        *,
        server_name: str | None = None,
    ) -> list[BaseTool]:
        """Retrieve LangChain-compatible tools from connected MCP servers.

        Each call creates a new session to the MCP server(s), retrieves the
        available tools, and returns them.  The tools can then be used with
        ``tool.ainvoke(...)`` in agent nodes.

        Args:
            server_name: Optional server name to restrict tool retrieval to
                a single server.  If ``None``, tools from **all** configured
                servers are returned.

        Returns:
            A list of LangChain ``BaseTool`` instances exposed by the
            MCP server(s).

        Raises:
            ConnectionError: If the MCP server cannot be reached.
            ValueError: If *server_name* is not found in configured
                connections.
        """
        if server_name and server_name not in self._connections:
            raise ValueError(
                f"Unknown MCP server: {server_name!r}. "
                f"Available: {sorted(self._connections.keys())}"
            )

        try:
            tools = await self.client.get_tools(server_name=server_name)
            logger.info(
                "Retrieved %d tool(s)%s: %s",
                len(tools),
                f" from {server_name!r}" if server_name else "",
                ", ".join(t.name for t in tools),
            )
            return tools
        except Exception as exc:
            logger.error("Failed to retrieve MCP tools: %s", exc)
            raise ConnectionError(
                f"Failed to connect to MCP server: {exc}"
            ) from exc

    def get_server_names(self) -> list[str]:
        """Return the list of configured MCP server names.

        Returns:
            Sorted list of server name strings.
        """
        return sorted(self._connections.keys())
