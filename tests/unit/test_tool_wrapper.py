"""Unit tests for RAGToolWrapper (B2).

Verifies that ``RAGToolWrapper`` correctly:
- normalizes parameters before forwarding to MCP tools,
- parses various result formats (JSON string, dict, list) into domain objects,
- handles missing tools gracefully (returns empty fallback),
- handles tool invocation errors gracefully (returns empty fallback),
- supports optional ``collection`` filtering in ``search()``.

All tests use mocked MCP tools — no real server connections.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.state import RetrievedContext
from src.tools.tool_wrapper import (
    RAGToolWrapper,
    TOOL_GET_DOCUMENT_SUMMARY,
    TOOL_LIST_COLLECTIONS,
    TOOL_QUERY_KNOWLEDGE_HUB,
    _parse_collections,
    _parse_doc_info,
    _parse_search_results,
    _safe_json_loads,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, return_value: Any = None) -> MagicMock:
    """Create a mock BaseTool with given name and optional async return value."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _sample_search_results() -> list[dict[str, Any]]:
    """Return sample search result dicts mimicking RAG Server output."""
    return [
        {
            "content": "LoRA adds low-rank adapters to transformer weights.",
            "source": "LoRA: Low-Rank Adaptation",
            "doc_id": "doc_001",
            "relevance_score": 0.95,
            "chunk_index": 0,
        },
        {
            "content": "QLoRA combines quantization with LoRA.",
            "source": "QLoRA Paper",
            "doc_id": "doc_002",
            "relevance_score": 0.87,
            "chunk_index": 3,
        },
    ]


# ---------------------------------------------------------------------------
# _safe_json_loads
# ---------------------------------------------------------------------------


class TestSafeJsonLoads:
    """Tests for the JSON parsing helper."""

    def test_parses_json_string(self) -> None:
        """Valid JSON string should be deserialized."""
        raw = json.dumps({"key": "value"})
        assert _safe_json_loads(raw) == {"key": "value"}

    def test_returns_dict_as_is(self) -> None:
        """Dict input should be returned unchanged."""
        data = {"key": "value"}
        assert _safe_json_loads(data) is data

    def test_returns_list_as_is(self) -> None:
        """List input should be returned unchanged."""
        data = [1, 2, 3]
        assert _safe_json_loads(data) is data

    def test_invalid_json_string_returned_as_is(self) -> None:
        """Invalid JSON string should be returned as the raw string."""
        raw = "not valid json {{"
        assert _safe_json_loads(raw) == raw

    def test_returns_none_as_is(self) -> None:
        """None should pass through unchanged."""
        assert _safe_json_loads(None) is None

    def test_returns_int_as_is(self) -> None:
        """Integer should pass through unchanged."""
        assert _safe_json_loads(42) == 42


# ---------------------------------------------------------------------------
# _parse_search_results
# ---------------------------------------------------------------------------


class TestParseSearchResults:
    """Tests for search result parsing."""

    def test_parse_list_of_dicts(self) -> None:
        """A plain list of result dicts should parse correctly."""
        items = _sample_search_results()
        results = _parse_search_results(items)
        assert len(results) == 2
        assert all(isinstance(r, RetrievedContext) for r in results)
        assert results[0].doc_id == "doc_001"
        assert results[0].relevance_score == 0.95
        assert results[1].source == "QLoRA Paper"

    def test_parse_json_string(self) -> None:
        """A JSON-encoded list of dicts should be decoded and parsed."""
        items = _sample_search_results()
        raw = json.dumps(items)
        results = _parse_search_results(raw)
        assert len(results) == 2
        assert results[0].content == "LoRA adds low-rank adapters to transformer weights."

    def test_parse_wrapped_dict_with_results_key(self) -> None:
        """A dict wrapping results under 'results' key should unwrap."""
        items = _sample_search_results()
        wrapped = {"results": items, "total": 2}
        results = _parse_search_results(wrapped)
        assert len(results) == 2

    def test_parse_wrapped_dict_with_chunks_key(self) -> None:
        """A dict wrapping results under 'chunks' key should unwrap."""
        items = _sample_search_results()
        wrapped = {"chunks": items}
        results = _parse_search_results(wrapped)
        assert len(results) == 2

    def test_empty_list(self) -> None:
        """An empty list should return an empty result."""
        assert _parse_search_results([]) == []

    def test_unexpected_type_returns_empty(self) -> None:
        """Non-list/non-dict should return empty list."""
        assert _parse_search_results(42) == []
        assert _parse_search_results("plain text") == []

    def test_alternative_key_names(self) -> None:
        """Tolerates 'title' for 'source', 'document_id' for 'doc_id', 'score' for 'relevance_score'."""
        item = {
            "content": "Some text",
            "title": "Paper Title",
            "document_id": "alt_001",
            "score": 0.77,
        }
        results = _parse_search_results([item])
        assert len(results) == 1
        assert results[0].source == "Paper Title"
        assert results[0].doc_id == "alt_001"
        assert results[0].relevance_score == 0.77

    def test_skips_non_dict_items_in_list(self) -> None:
        """Non-dict items in the list should be silently skipped."""
        items = [
            {"content": "good", "source": "s", "doc_id": "d1", "relevance_score": 0.5},
            "bad item",
            42,
        ]
        results = _parse_search_results(items)
        assert len(results) == 1
        assert results[0].doc_id == "d1"

    def test_defaults_for_missing_fields(self) -> None:
        """Missing fields should get sensible defaults."""
        item: dict[str, Any] = {}
        results = _parse_search_results([item])
        assert len(results) == 1
        ctx = results[0]
        assert ctx.content == ""
        assert ctx.source == "unknown"
        assert ctx.doc_id == ""
        assert ctx.relevance_score == 0.0
        assert ctx.chunk_index is None


# ---------------------------------------------------------------------------
# _parse_collections
# ---------------------------------------------------------------------------


class TestParseCollections:
    """Tests for collection list parsing."""

    def test_parse_list_of_strings(self) -> None:
        """Plain list of strings should parse directly."""
        result = _parse_collections(["arxiv_papers", "textbooks"])
        assert result == ["arxiv_papers", "textbooks"]

    def test_parse_list_of_dicts(self) -> None:
        """List of dicts with 'name' key should extract names."""
        data = [
            {"name": "arxiv_papers", "count": 42},
            {"name": "textbooks", "count": 10},
        ]
        result = _parse_collections(data)
        assert result == ["arxiv_papers", "textbooks"]

    def test_parse_list_of_dicts_collection_key(self) -> None:
        """Dicts with 'collection' key (fallback) should also work."""
        data = [{"collection": "my_col"}]
        result = _parse_collections(data)
        assert result == ["my_col"]

    def test_parse_wrapped_dict(self) -> None:
        """A dict with 'collections' key should unwrap."""
        data = {"collections": ["a", "b"]}
        result = _parse_collections(data)
        assert result == ["a", "b"]

    def test_parse_json_string(self) -> None:
        """JSON string should be decoded first."""
        raw = json.dumps(["col_a", "col_b"])
        result = _parse_collections(raw)
        assert result == ["col_a", "col_b"]

    def test_empty_list(self) -> None:
        """Empty list should return empty."""
        assert _parse_collections([]) == []

    def test_unexpected_type_returns_empty(self) -> None:
        """Non-list/non-dict should return empty list."""
        assert _parse_collections(42) == []

    def test_skips_empty_name_dicts(self) -> None:
        """Dicts with no name/collection key should be skipped."""
        data = [{"name": "good"}, {"something_else": "bad"}]
        result = _parse_collections(data)
        assert result == ["good"]


# ---------------------------------------------------------------------------
# _parse_doc_info
# ---------------------------------------------------------------------------


class TestParseDocInfo:
    """Tests for document info parsing."""

    def test_parse_dict(self) -> None:
        """Dict input should be returned as-is."""
        data = {"title": "LoRA Paper", "summary": "Low-rank adaptation..."}
        result = _parse_doc_info(data)
        assert result == data

    def test_parse_json_string(self) -> None:
        """JSON string should be decoded."""
        data = {"title": "Paper", "pages": 12}
        result = _parse_doc_info(json.dumps(data))
        assert result == data

    def test_unexpected_type_returns_empty_dict(self) -> None:
        """Non-dict type should return empty dict."""
        assert _parse_doc_info(42) == {}
        assert _parse_doc_info("plain text") == {}


# ---------------------------------------------------------------------------
# RAGToolWrapper — initialization & helpers
# ---------------------------------------------------------------------------


class TestRAGToolWrapperInit:
    """Tests for wrapper initialization and utility methods."""

    def test_init_indexes_tools_by_name(self) -> None:
        """Tools should be indexed by their .name attribute."""
        tools = [
            _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB),
            _make_mock_tool(TOOL_LIST_COLLECTIONS),
        ]
        wrapper = RAGToolWrapper(tools)
        assert wrapper.has_tool(TOOL_QUERY_KNOWLEDGE_HUB)
        assert wrapper.has_tool(TOOL_LIST_COLLECTIONS)
        assert not wrapper.has_tool("nonexistent")

    def test_available_tools_returns_sorted(self) -> None:
        """available_tools() should return sorted names."""
        tools = [
            _make_mock_tool("z_tool"),
            _make_mock_tool("a_tool"),
        ]
        wrapper = RAGToolWrapper(tools)
        assert wrapper.available_tools() == ["a_tool", "z_tool"]

    def test_init_with_empty_list(self) -> None:
        """Empty tool list should be handled gracefully."""
        wrapper = RAGToolWrapper([])
        assert wrapper.available_tools() == []
        assert not wrapper.has_tool(TOOL_QUERY_KNOWLEDGE_HUB)


# ---------------------------------------------------------------------------
# RAGToolWrapper.search()
# ---------------------------------------------------------------------------


class TestRAGToolWrapperSearch:
    """Tests for the search() method."""

    async def test_search_basic_invocation(self) -> None:
        """search() should invoke query_knowledge_hub with correct params."""
        items = _sample_search_results()
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB, return_value=items)
        wrapper = RAGToolWrapper([tool])

        results = await wrapper.search("What is LoRA?", top_k=3)

        tool.ainvoke.assert_awaited_once_with({"query": "What is LoRA?", "top_k": 3})
        assert len(results) == 2
        assert all(isinstance(r, RetrievedContext) for r in results)

    async def test_search_with_collection_param(self) -> None:
        """search() with collection should include it in the params."""
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB, return_value=[])
        wrapper = RAGToolWrapper([tool])

        await wrapper.search("query", top_k=5, collection="arxiv")

        call_args = tool.ainvoke.call_args[0][0]
        assert call_args["collection"] == "arxiv"
        assert call_args["query"] == "query"
        assert call_args["top_k"] == 5

    async def test_search_without_collection_omits_key(self) -> None:
        """search() without collection should NOT include it in params."""
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB, return_value=[])
        wrapper = RAGToolWrapper([tool])

        await wrapper.search("query")

        call_args = tool.ainvoke.call_args[0][0]
        assert "collection" not in call_args

    async def test_search_parses_json_string_result(self) -> None:
        """search() should handle JSON-string results from MCP."""
        items = _sample_search_results()
        raw = json.dumps(items)
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB, return_value=raw)
        wrapper = RAGToolWrapper([tool])

        results = await wrapper.search("test query")

        assert len(results) == 2
        assert results[0].source == "LoRA: Low-Rank Adaptation"

    async def test_search_missing_tool_returns_empty(self) -> None:
        """search() should return [] when tool is not available."""
        wrapper = RAGToolWrapper([])
        results = await wrapper.search("anything")
        assert results == []

    async def test_search_invocation_error_returns_empty(self) -> None:
        """search() should catch exceptions and return []."""
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB)
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("network error"))
        wrapper = RAGToolWrapper([tool])

        results = await wrapper.search("What is LoRA?")

        assert results == []

    async def test_search_default_top_k_is_five(self) -> None:
        """search() default top_k should be 5."""
        tool = _make_mock_tool(TOOL_QUERY_KNOWLEDGE_HUB, return_value=[])
        wrapper = RAGToolWrapper([tool])

        await wrapper.search("q")

        call_args = tool.ainvoke.call_args[0][0]
        assert call_args["top_k"] == 5


# ---------------------------------------------------------------------------
# RAGToolWrapper.list_collections()
# ---------------------------------------------------------------------------


class TestRAGToolWrapperListCollections:
    """Tests for the list_collections() method."""

    async def test_list_collections_basic(self) -> None:
        """list_collections() should invoke tool and parse result."""
        raw = ["arxiv_papers", "textbooks"]
        tool = _make_mock_tool(TOOL_LIST_COLLECTIONS, return_value=raw)
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.list_collections()

        tool.ainvoke.assert_awaited_once_with({"include_stats": True})
        assert result == ["arxiv_papers", "textbooks"]

    async def test_list_collections_include_stats_false(self) -> None:
        """list_collections(include_stats=False) should forward param."""
        tool = _make_mock_tool(TOOL_LIST_COLLECTIONS, return_value=[])
        wrapper = RAGToolWrapper([tool])

        await wrapper.list_collections(include_stats=False)

        call_args = tool.ainvoke.call_args[0][0]
        assert call_args["include_stats"] is False

    async def test_list_collections_missing_tool_returns_empty(self) -> None:
        """list_collections() should return [] when tool unavailable."""
        wrapper = RAGToolWrapper([])
        result = await wrapper.list_collections()
        assert result == []

    async def test_list_collections_invocation_error_returns_empty(self) -> None:
        """list_collections() should catch exceptions and return []."""
        tool = _make_mock_tool(TOOL_LIST_COLLECTIONS)
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.list_collections()

        assert result == []

    async def test_list_collections_parses_json_string(self) -> None:
        """list_collections() should handle JSON-string results."""
        raw = json.dumps(["col1", "col2"])
        tool = _make_mock_tool(TOOL_LIST_COLLECTIONS, return_value=raw)
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.list_collections()
        assert result == ["col1", "col2"]


# ---------------------------------------------------------------------------
# RAGToolWrapper.get_doc_info()
# ---------------------------------------------------------------------------


class TestRAGToolWrapperGetDocInfo:
    """Tests for the get_doc_info() method."""

    async def test_get_doc_info_basic(self) -> None:
        """get_doc_info() should invoke tool with doc_id and parse result."""
        doc_data = {"title": "LoRA Paper", "summary": "Low-rank adaptation", "pages": 12}
        tool = _make_mock_tool(TOOL_GET_DOCUMENT_SUMMARY, return_value=doc_data)
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.get_doc_info("doc_001")

        tool.ainvoke.assert_awaited_once_with({"doc_id": "doc_001"})
        assert result == doc_data

    async def test_get_doc_info_missing_tool_returns_empty_dict(self) -> None:
        """get_doc_info() should return {} when tool unavailable."""
        wrapper = RAGToolWrapper([])
        result = await wrapper.get_doc_info("doc_001")
        assert result == {}

    async def test_get_doc_info_invocation_error_returns_empty_dict(self) -> None:
        """get_doc_info() should catch exceptions and return {}."""
        tool = _make_mock_tool(TOOL_GET_DOCUMENT_SUMMARY)
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("server error"))
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.get_doc_info("doc_001")

        assert result == {}

    async def test_get_doc_info_parses_json_string(self) -> None:
        """get_doc_info() should handle JSON-string result."""
        doc_data = {"title": "Paper X", "author": "Smith"}
        raw = json.dumps(doc_data)
        tool = _make_mock_tool(TOOL_GET_DOCUMENT_SUMMARY, return_value=raw)
        wrapper = RAGToolWrapper([tool])

        result = await wrapper.get_doc_info("doc_x")
        assert result == doc_data
