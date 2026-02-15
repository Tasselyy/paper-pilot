"""RAGToolWrapper: unified interface for RAG MCP tool invocations.

Wraps the raw MCP tools (``query_knowledge_hub``, ``list_collections``,
``get_document_summary``) exposed by the RAG Server and provides:

* **Parameter normalization** — callers use Python-friendly arguments;
  the wrapper maps them to the exact parameter names each MCP tool expects.
* **Result parsing** — raw MCP return values (strings / dicts / lists) are
  parsed into typed domain objects (``RetrievedContext``) or plain Python
  collections.
* **Error handling** — tool invocation failures are logged and produce
  graceful fallback values (empty list / empty dict) instead of propagating
  exceptions to the agent graph.

Design reference: PAPER_PILOT_DESIGN.md §7.2, DEV_SPEC §3.2 / §5.3.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool

from src.agent.state import RetrievedContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known MCP tool names (must match the tools exposed by the RAG Server)
# ---------------------------------------------------------------------------

TOOL_QUERY_KNOWLEDGE_HUB = "query_knowledge_hub"
TOOL_LIST_COLLECTIONS = "list_collections"
TOOL_GET_DOCUMENT_SUMMARY = "get_document_summary"

# ---------------------------------------------------------------------------
# Result parsers (private helpers)
# ---------------------------------------------------------------------------


def _safe_json_loads(raw: Any) -> Any:
    """Attempt to JSON-decode *raw* if it is a string; return as-is otherwise.

    Args:
        raw: Raw MCP tool output — may be a JSON string, dict, list, or other.

    Returns:
        Parsed Python object, or the original value if parsing is not needed
        or fails.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _parse_single_context(item: dict[str, Any]) -> RetrievedContext:
    """Convert a single raw result dict into a ``RetrievedContext``.

    The function tolerates common key-name variations returned by different
    RAG Server implementations (e.g. ``score`` vs ``relevance_score``).

    Args:
        item: A single search-result dict from MCP.

    Returns:
        A validated ``RetrievedContext`` instance.
    """
    return RetrievedContext(
        content=str(item.get("content", "")),
        source=str(item.get("source", item.get("title", "unknown"))),
        doc_id=str(item.get("doc_id", item.get("document_id", ""))),
        relevance_score=float(item.get("relevance_score", item.get("score", 0.0))),
        chunk_index=item.get("chunk_index"),
    )


def _parse_search_results(raw: Any) -> list[RetrievedContext]:
    """Parse the raw output of ``query_knowledge_hub`` into domain objects.

    Args:
        raw: Raw return value from the MCP tool (string or already-parsed).

    Returns:
        A list of ``RetrievedContext`` instances.  Returns an empty list if
        the result cannot be parsed.
    """
    data = _safe_json_loads(raw)

    # If the result is a dict with a "results" key, unwrap it.
    if isinstance(data, dict):
        data = data.get("results", data.get("chunks", []))

    if not isinstance(data, list):
        logger.warning(
            "Unexpected search result type %s — returning empty list",
            type(data).__name__,
        )
        return []

    contexts: list[RetrievedContext] = []
    for item in data:
        if isinstance(item, dict):
            try:
                contexts.append(_parse_single_context(item))
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping unparseable search result item: %s", exc)
        else:
            logger.debug("Skipping non-dict search result item: %r", item)
    return contexts


def _parse_collections(raw: Any) -> list[str]:
    """Parse the raw output of ``list_collections`` into collection names.

    Args:
        raw: Raw return value from the MCP tool.

    Returns:
        A list of collection name strings.  Empty list on parse failure.
    """
    data = _safe_json_loads(raw)

    # If the result is a dict with a "collections" key, unwrap it.
    if isinstance(data, dict):
        data = data.get("collections", [])

    if not isinstance(data, list):
        logger.warning(
            "Unexpected collections result type %s — returning empty list",
            type(data).__name__,
        )
        return []

    names: list[str] = []
    for item in data:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            # e.g. {"name": "arxiv_papers", "count": 42}
            name = item.get("name", item.get("collection", ""))
            if name:
                names.append(str(name))
    return names


def _parse_doc_info(raw: Any) -> dict[str, Any]:
    """Parse the raw output of ``get_document_summary``.

    Args:
        raw: Raw return value from the MCP tool.

    Returns:
        A dict with document metadata.  Empty dict on parse failure.
    """
    data = _safe_json_loads(raw)
    if isinstance(data, dict):
        return data
    logger.warning(
        "Unexpected doc info result type %s — returning empty dict",
        type(data).__name__,
    )
    return {}


# ---------------------------------------------------------------------------
# RAGToolWrapper
# ---------------------------------------------------------------------------


class RAGToolWrapper:
    """Unified interface for RAG Server tool invocations via MCP.

    Accepts a list of LangChain ``BaseTool`` instances (typically obtained
    from ``MCPClientManager.get_tools()``) and provides typed, error-handled
    async methods for each RAG operation.

    Attributes:
        tools: Mapping from tool name to ``BaseTool`` instance.

    Example::

        tools = await mcp_manager.get_tools()
        rag = RAGToolWrapper(tools)
        results = await rag.search("What is LoRA?", top_k=5)
    """

    def __init__(self, mcp_tools: list[BaseTool]) -> None:
        """Initialize with a list of MCP tools.

        Args:
            mcp_tools: LangChain ``BaseTool`` instances from the MCP client.
        """
        self.tools: dict[str, BaseTool] = {t.name: t for t in mcp_tools}
        logger.info(
            "RAGToolWrapper initialized with %d tool(s): %s",
            len(self.tools),
            ", ".join(sorted(self.tools.keys())),
        )

    # -- public helpers -----------------------------------------------------

    def has_tool(self, name: str) -> bool:
        """Check whether a tool with the given name is available.

        Args:
            name: MCP tool name.

        Returns:
            ``True`` if the tool was provided during initialization.
        """
        return name in self.tools

    def available_tools(self) -> list[str]:
        """Return sorted list of available tool names.

        Returns:
            Sorted list of tool name strings.
        """
        return sorted(self.tools.keys())

    # -- RAG operations -----------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[RetrievedContext]:
        """Search the knowledge base via ``query_knowledge_hub``.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            collection: Optional collection name to restrict the search to.

        Returns:
            A list of ``RetrievedContext`` objects parsed from the raw
            MCP result.  Returns an empty list if the tool is unavailable
            or the invocation fails (graceful degradation).
        """
        tool = self.tools.get(TOOL_QUERY_KNOWLEDGE_HUB)
        if tool is None:
            logger.warning(
                "Tool %r not available — returning empty results",
                TOOL_QUERY_KNOWLEDGE_HUB,
            )
            return []

        params: dict[str, Any] = {"query": query, "top_k": top_k}
        if collection:
            params["collection"] = collection

        try:
            raw = await tool.ainvoke(params)
            results = _parse_search_results(raw)
            logger.info(
                "RAG search returned %d result(s) for query=%r (top_k=%d)",
                len(results),
                query[:80],
                top_k,
            )
            return results
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)
            return []

    async def list_collections(
        self,
        *,
        include_stats: bool = True,
    ) -> list[str]:
        """List available knowledge collections via ``list_collections``.

        Args:
            include_stats: Whether to request collection statistics from
                the MCP server.

        Returns:
            A list of collection name strings.  Returns an empty list if
            the tool is unavailable or invocation fails.
        """
        tool = self.tools.get(TOOL_LIST_COLLECTIONS)
        if tool is None:
            logger.warning(
                "Tool %r not available — returning empty list",
                TOOL_LIST_COLLECTIONS,
            )
            return []

        params: dict[str, Any] = {"include_stats": include_stats}

        try:
            raw = await tool.ainvoke(params)
            collections = _parse_collections(raw)
            logger.info(
                "Listed %d collection(s): %s",
                len(collections),
                ", ".join(collections) if collections else "(none)",
            )
            return collections
        except Exception as exc:
            logger.warning("list_collections failed: %s", exc)
            return []

    async def get_doc_info(self, doc_id: str) -> dict[str, Any]:
        """Retrieve document summary via ``get_document_summary``.

        Args:
            doc_id: Unique document identifier.

        Returns:
            A dict containing document metadata (title, summary, etc.).
            Returns an empty dict if the tool is unavailable or invocation
            fails.
        """
        tool = self.tools.get(TOOL_GET_DOCUMENT_SUMMARY)
        if tool is None:
            logger.warning(
                "Tool %r not available — returning empty dict",
                TOOL_GET_DOCUMENT_SUMMARY,
            )
            return {}

        params: dict[str, Any] = {"doc_id": doc_id}

        try:
            raw = await tool.ainvoke(params)
            info = _parse_doc_info(raw)
            logger.info(
                "Retrieved doc info for doc_id=%r: %d field(s)",
                doc_id,
                len(info),
            )
            return info
        except Exception as exc:
            logger.warning("get_doc_info failed for doc_id=%r: %s", doc_id, exc)
            return {}
