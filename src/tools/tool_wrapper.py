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

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.agent.state import RetrievedContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known MCP tool names (must match the tools exposed by the RAG Server)
# ---------------------------------------------------------------------------

TOOL_QUERY_KNOWLEDGE_HUB = "query_knowledge_hub"
TOOL_LIST_COLLECTIONS = "list_collections"
TOOL_GET_DOCUMENT_SUMMARY = "get_document_summary"

# Timeout for a single RAG search to avoid hanging (e.g. MCP stdio or slow server).
RAG_SEARCH_TIMEOUT_SEC = 90

# ---------------------------------------------------------------------------
# Wrapper for query_knowledge_hub to avoid RecursionError in LangChain
# ---------------------------------------------------------------------------
# The MCP adapter's tool can have an args_schema that causes
# get_all_basemodel_annotations() to recurse infinitely. We wrap it with a
# tool that has a flat Pydantic schema and call the underlying _arun directly.


class _QueryKnowledgeHubInput(BaseModel):
    """Simple input schema for query_knowledge_hub (avoids MCP schema recursion)."""

    query: str = Field(description="Natural-language search query")
    top_k: int = Field(default=5, description="Maximum number of results")
    collection: str | None = Field(default=None, description="Optional collection name")


def _wrap_query_knowledge_hub_tool(original: BaseTool) -> BaseTool:
    """Wrap the MCP query_knowledge_hub tool with a simple-schema tool.

    Calls the original tool's _arun directly to skip its args_schema introspection.
    """
    # Use a closure to hold the reference to the original tool
    _real = original

    class _QueryKnowledgeHubWrapper(BaseTool):
        name: str = TOOL_QUERY_KNOWLEDGE_HUB
        description: str = getattr(
            original, "description", "Search the knowledge base (hybrid search)."
        )
        args_schema: type[BaseModel] = _QueryKnowledgeHubInput

        def _run(self, query: str, top_k: int = 5, collection: str | None = None) -> Any:
            raise NotImplementedError("Use async ainvoke for query_knowledge_hub")

        async def _arun(
            self,
            query: str,
            top_k: int = 5,
            collection: str | None = None,
            *,
            config: RunnableConfig | None = None,
            run_manager: AsyncCallbackManagerForToolRun | None = None,
            **kwargs: Any,
        ) -> Any:
            # Call the underlying MCP tool's _arun to bypass its args_schema.
            # StructuredTool._arun requires config (and optionally run_manager);
            # pass them through so ainvoke(config=...) works.
            params: dict[str, Any] = {"query": query, "top_k": top_k}
            if collection is not None:
                params["collection"] = collection
            return await _real._arun(
                **params,
                config=config if config is not None else {},
                run_manager=run_manager,
                **kwargs,
            )

    return _QueryKnowledgeHubWrapper()

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
        text = raw.strip()
        if not text:
            return text

        # 1) Plain JSON payload
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2) Markdown fenced JSON payload (common in MCP text responses)
        fenced_blocks = re.findall(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        )
        for block in fenced_blocks:
            candidate = block.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

        # 3) JSON embedded within surrounding prose
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except (json.JSONDecodeError, ValueError):
                continue

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


def _parse_citation_context(citation: dict[str, Any]) -> RetrievedContext:
    """Convert a citation dict into ``RetrievedContext``.

    Supports citation shapes like:
    - {"source": "...", "score": 0.1, "text_snippet": "...", "chunk_id": "...", "metadata": {...}}
    """
    metadata = citation.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    chunk_index = metadata_dict.get("chunk_index")
    doc_id = citation.get("doc_id") or citation.get("chunk_id") or citation.get("document_id", "")
    text = (
        citation.get("content")
        or citation.get("text")
        or citation.get("text_snippet")
        or ""
    )
    return RetrievedContext(
        content=str(text),
        source=str(citation.get("source", citation.get("title", "unknown"))),
        doc_id=str(doc_id),
        relevance_score=float(citation.get("relevance_score", citation.get("score", 0.0))),
        chunk_index=chunk_index if isinstance(chunk_index, int) else None,
    )


def _contexts_from_citations(citations: Any) -> list[RetrievedContext]:
    """Build ``RetrievedContext`` list from citation arrays."""
    if not isinstance(citations, list):
        return []
    contexts: list[RetrievedContext] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        try:
            contexts.append(_parse_citation_context(citation))
        except (ValueError, TypeError):
            continue
    return contexts


def _unwrap_content_and_artifact(raw: Any) -> Any:
    """Unwrap langchain-mcp-adapters (content, artifact) tuple into parseable payload.

    When the adapter uses response_format='content_and_artifact', ainvoke returns
    (content, artifact). Prefer artifact.structured_content if present; else
    extract text from content blocks and parse as JSON.
    """
    if not isinstance(raw, tuple) or len(raw) == 0:
        return raw
    content = raw[0]
    artifact = raw[1] if len(raw) > 1 else None
    if artifact is not None and hasattr(artifact, "structured_content"):
        sc = artifact.structured_content
        if isinstance(sc, (dict, list)):
            return sc
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    texts.append(t)
            elif isinstance(block, str):
                texts.append(block)
        if texts:
            return _safe_json_loads(texts[0]) if len(texts) == 1 else _safe_json_loads("\n".join(texts))
    return content


def _parse_search_results(raw: Any) -> list[RetrievedContext]:
    """Parse the raw output of ``query_knowledge_hub`` into domain objects.

    Args:
        raw: Raw return value from the MCP tool — may be (content, artifact)
            tuple from the adapter, or string/dict/list.

    Returns:
        A list of ``RetrievedContext`` instances.  Returns an empty list if
        the result cannot be parsed.
    """
    payload = _unwrap_content_and_artifact(raw)
    data = payload if isinstance(payload, (dict, list)) else _safe_json_loads(payload)

    # If the result is a dict with known wrapper keys, unwrap it.
    if isinstance(data, dict):
        if "result" in data:
            # FastMCP streamable-http often wraps tool return under "result".
            result_payload = data["result"]
            if isinstance(result_payload, str):
                result_payload = _safe_json_loads(result_payload)
            if isinstance(result_payload, dict) and "citations" in result_payload:
                contexts = _contexts_from_citations(result_payload.get("citations"))
                if contexts:
                    return contexts
            if isinstance(result_payload, dict) and "results" in result_payload:
                data = result_payload["results"]
            else:
                data = result_payload
        elif "results" in data:
            data = data["results"]
        elif "chunks" in data:
            data = data["chunks"]
        elif "citations" in data:
            contexts = _contexts_from_citations(data.get("citations"))
            if contexts:
                return contexts
            data = []
        elif "data" in data and isinstance(data["data"], list):
            data = data["data"]
        elif "content" in data:
            # Single result item represented as one dict instead of list
            data = [data]
        else:
            data = []

    # Some MCP servers return plain text; preserve it as a fallback context.
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        lowered = text.lower()
        if (
            "no result" in lowered
            or "no relevant" in lowered
            or "not found" in lowered
            or "0 result" in lowered
        ):
            logger.info("RAG search returned no matches text payload")
            return []
        return [
            RetrievedContext(
                content=text,
                source=TOOL_QUERY_KNOWLEDGE_HUB,
                doc_id="",
                relevance_score=0.0,
                chunk_index=None,
            )
        ]

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
        self.tools = {t.name: t for t in mcp_tools}
        # Wrap query_knowledge_hub to avoid RecursionError in LangChain's
        # args_schema introspection (MCP tool schema can cause infinite recursion).
        if TOOL_QUERY_KNOWLEDGE_HUB in self.tools:
            self.tools[TOOL_QUERY_KNOWLEDGE_HUB] = _wrap_query_knowledge_hub_tool(
                self.tools[TOOL_QUERY_KNOWLEDGE_HUB],
            )
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
            logger.debug("RAG search with collection=%r", collection)

        logger.info(
            "RAG search starting (timeout=%ds) query=%r",
            RAG_SEARCH_TIMEOUT_SEC,
            query[:80],
        )
        async def _invoke_with_timeout(timeout_sec: int) -> list[RetrievedContext]:
            raw = await asyncio.wait_for(
                tool.ainvoke(params),
                timeout=timeout_sec,
            )
            return _parse_search_results(raw)

        try:
            results = await _invoke_with_timeout(RAG_SEARCH_TIMEOUT_SEC)
            logger.info(
                "RAG search returned %d result(s) for query=%r (top_k=%d)",
                len(results),
                query[:80],
                top_k,
            )
            return results
        except asyncio.TimeoutError:
            extended_timeout = RAG_SEARCH_TIMEOUT_SEC * 2
            logger.warning(
                "RAG search timed out after %ds for query=%r; retrying once with timeout=%ds",
                RAG_SEARCH_TIMEOUT_SEC,
                query[:80],
                extended_timeout,
            )
            try:
                results = await _invoke_with_timeout(extended_timeout)
                logger.info(
                    "RAG search retry returned %d result(s) for query=%r (top_k=%d)",
                    len(results),
                    query[:80],
                    top_k,
                )
                return results
            except asyncio.TimeoutError:
                logger.warning(
                    "RAG search timed out again after %ds for query=%r",
                    extended_timeout,
                    query[:80],
                )
                return []
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
