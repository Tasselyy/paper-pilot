"""MCP Client: connect to RAG Server via stdio or streamable-http.

Wraps ``langchain-mcp-adapters``'s ``MultiServerMCPClient`` and provides a
config-driven ``MCPClientManager`` that reads ``MCPConfig`` from settings,
builds connection dicts, and exposes ``get_tools()`` to retrieve
LangChain-compatible tools for the agent graph.

When using stdio transport, the MCP subprocess's stderr can be redirected via
``mcp_stderr`` config (e.g. ``devnull`` or a log file path) to avoid the
subprocess blocking when the parent does not consume stderr (pipe buffer full).

Design reference: DEV_SPEC §3.2, §5.3 (tools/mcp_client.py), §7.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.config import MCPConfig, MCPConnectionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patch stdio session to pass errlog (avoid subprocess stderr pipe blocking)
# ---------------------------------------------------------------------------

_STDIO_SERVER_PARAM_KEYS = frozenset(
    {"command", "args", "env", "cwd", "encoding", "encoding_error_handler"}
)


def _apply_stdio_stderr_patch() -> None:
    """Patch langchain_mcp_adapters so stdio sessions can redirect stderr.

    When mcp_stderr is set in connection config we pass errlog to the MCP SDK's
    stdio_client; otherwise the subprocess writes to a pipe. If the parent never
    reads that pipe, the buffer fills and the server blocks before writing the
    JSON-RPC response to stdout, so the client appears stuck.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from langchain_mcp_adapters import sessions as _sessions

    @asynccontextmanager
    async def _patched_create_stdio_session(**kwargs: Any) -> Any:
        stderr_opt = kwargs.pop("_paper_pilot_stderr", None)
        if stderr_opt == "devnull":
            errlog = open(os.devnull, "w", encoding="utf-8")
        elif isinstance(stderr_opt, str) and stderr_opt.strip():
            path = Path(stderr_opt.strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            errlog = open(path, "w", encoding="utf-8")
        else:
            errlog = sys.stderr

        session_kwargs = kwargs.pop("session_kwargs", None)
        server_params = StdioServerParameters(
            **{k: v for k, v in kwargs.items() if k in _STDIO_SERVER_PARAM_KEYS}
        )
        try:
            async with stdio_client(server_params, errlog=errlog) as (read, write):
                async with ClientSession(
                    read, write, **(session_kwargs or {})
                ) as session:
                    yield session
        finally:
            if errlog is not sys.stderr and not errlog.closed:
                errlog.close()

    _sessions._create_stdio_session = _patched_create_stdio_session
    logger.debug("Applied stdio stderr redirect patch for MCP sessions")


# Apply once at import so any connection with _paper_pilot_stderr uses it
_apply_stdio_stderr_patch()

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
            # Avoid stdout/stderr buffering so MCP JSON-RPC responses flush immediately
            env["PYTHONUNBUFFERED"] = "1"
            out["env"] = env
        # Redirect subprocess stderr to avoid pipe blocking (RAG server logs to stderr;
        # if we use PIPE and never read, the buffer fills and the process blocks).
        if getattr(conn, "mcp_stderr", None):
            out["_paper_pilot_stderr"] = conn.mcp_stderr
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

    @asynccontextmanager
    async def session(self, server_name: str):
        """Open a long-lived MCP session for the given server.

        Use this when you want tools to reuse one process (e.g. one RAG
        stdio subprocess) for the whole run instead of spawning a new
        process per tool call. Enter the context before building the
        graph and keep it open until the run finishes.

        Args:
            server_name: Name of the server (e.g. ``"rag_server"``).

        Yields:
            An initialized MCP ``ClientSession``.

        Example::

            async with mcp_manager.session("rag_server") as session:
                tools = await mcp_manager.get_tools_using_session(session, "rag_server")
                rag = RAGToolWrapper(tools)
                # ... build graph, run agent; all RAG calls use this session
        """
        if server_name not in self._connections:
            raise ValueError(
                f"Unknown MCP server: {server_name!r}. "
                f"Available: {sorted(self._connections.keys())}"
            )
        async with self.client.session(server_name) as session:
            yield session

    async def get_tools_using_session(
        self,
        session: Any,
        server_name: str,
    ) -> list[BaseTool]:
        """Load tools that use an existing MCP session (no new process per call).

        Call this inside ``async with manager.session(server_name) as session:``
        so that all tool invocations use the same session (same RAG process).

        Args:
            session: The open ``ClientSession`` from ``manager.session(server_name)``.
            server_name: Same server name used to open the session.

        Returns:
            List of LangChain tools bound to the given session.
        """
        tools = await load_mcp_tools(
            session,
            connection=self._connections[server_name],
            callbacks=self.client.callbacks,
            server_name=server_name,
            tool_interceptors=self.client.tool_interceptors,
            tool_name_prefix=self.client.tool_name_prefix,
        )
        logger.info(
            "Retrieved %d tool(s) from %r (long-lived session): %s",
            len(tools),
            server_name,
            ", ".join(t.name for t in tools),
        )
        return tools
