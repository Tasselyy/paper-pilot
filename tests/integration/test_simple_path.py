"""Integration tests for the simple strategy path (B4).

Verifies that the full graph executes the simple path from START to END:

    START → load_memory → route → slot_fill → simple → critic
          → save_memory → format_output → END

All external dependencies (RAG MCP, Cloud LLM) are mocked.  The test
confirms that:
- The graph compiles and invokes end-to-end without errors.
- ``final_answer`` is non-empty and contains source citations.
- ``retrieved_contexts`` is populated from the (mocked) RAG search.
- ``intent`` is set by the router placeholder (factual).
- ``critic_verdict`` is set by the critic placeholder (passed=True).
- ``draft_answer`` is written by the simple strategy.
- State flows through all nodes in the correct order.

Design reference: DEV_SPEC B4.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.graph import build_main_graph
from src.agent.state import RetrievedContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sample_contexts() -> list[RetrievedContext]:
    """Return sample RetrievedContext objects mimicking RAG search output."""
    return [
        RetrievedContext(
            content="LoRA adds low-rank adapters to frozen pre-trained weights.",
            source="LoRA: Low-Rank Adaptation of Large Language Models",
            doc_id="doc_lora_001",
            relevance_score=0.95,
            chunk_index=0,
        ),
        RetrievedContext(
            content="QLoRA combines 4-bit quantization with LoRA for memory-efficient fine-tuning.",
            source="QLoRA: Efficient Finetuning of Quantized LLMs",
            doc_id="doc_qlora_002",
            relevance_score=0.88,
            chunk_index=1,
        ),
        RetrievedContext(
            content="Adapter layers inserted in parallel achieve comparable performance to full fine-tuning.",
            source="LoRA: Low-Rank Adaptation of Large Language Models",
            doc_id="doc_lora_001",
            relevance_score=0.82,
            chunk_index=5,
        ),
    ]


def _make_mock_rag() -> MagicMock:
    """Create a mock ``RAGToolWrapper`` that returns sample contexts."""
    rag = MagicMock()
    rag.search = AsyncMock(return_value=_sample_contexts())
    return rag


def _make_mock_llm(
    response_text: str = (
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning "
        "technique that inserts trainable low-rank decomposition matrices "
        "into transformer layers while keeping the original weights frozen. "
        "This approach drastically reduces the number of trainable parameters."
    ),
) -> MagicMock:
    """Create a mock ``BaseChatModel`` that returns a fixed response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSimplePath:
    """End-to-end integration tests for the simple strategy path."""

    @pytest.fixture()
    def mock_rag(self) -> MagicMock:
        """Provide a mocked RAGToolWrapper."""
        return _make_mock_rag()

    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        """Provide a mocked BaseChatModel."""
        return _make_mock_llm()

    @pytest.fixture()
    def graph(self, mock_rag: MagicMock, mock_llm: MagicMock):
        """Build a compiled graph with mocked RAG and LLM dependencies."""
        return build_main_graph(rag=mock_rag, llm=mock_llm)

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        """Provide a graph invocation config with a unique thread_id."""
        return {"configurable": {"thread_id": "test-simple-path"}}

    # -- Core assertions ---------------------------------------------------

    async def test_simple_path_completes_start_to_end(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """The graph should complete the full simple path without errors."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        assert result is not None

    async def test_final_answer_is_non_empty(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``final_answer`` must be non-empty after completing the simple path."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        final_answer = (
            result.get("final_answer")
            if isinstance(result, dict)
            else result.final_answer
        )
        assert final_answer, "final_answer should be non-empty"
        assert len(final_answer) > 10, "final_answer should be a meaningful string"

    async def test_final_answer_contains_source_citations(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``final_answer`` should contain a Sources section with paper names."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        final_answer = (
            result.get("final_answer")
            if isinstance(result, dict)
            else result.final_answer
        )
        assert "Sources" in final_answer, "final_answer should include Sources section"
        assert "LoRA" in final_answer, "Sources should reference LoRA paper"

    async def test_retrieved_contexts_populated(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``retrieved_contexts`` should contain the mocked RAG results."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        contexts = (
            result.get("retrieved_contexts", [])
            if isinstance(result, dict)
            else result.retrieved_contexts
        )
        assert len(contexts) == 3, "Should have 3 retrieved contexts from mock"
        assert all(
            isinstance(ctx, RetrievedContext) for ctx in contexts
        ), "All contexts should be RetrievedContext instances"

    async def test_draft_answer_written_by_simple_strategy(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``draft_answer`` should be set by the simple strategy node."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        draft = (
            result.get("draft_answer", "")
            if isinstance(result, dict)
            else result.draft_answer
        )
        assert draft, "draft_answer should be non-empty"
        assert "LoRA" in draft, "draft_answer should mention LoRA (from mock LLM)"

    async def test_intent_set_by_router(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``intent`` should be set by the router placeholder (factual type)."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        intent = (
            result.get("intent")
            if isinstance(result, dict)
            else result.intent
        )
        assert intent is not None, "intent should be set by router"
        assert intent.type == "factual", "Router placeholder should set factual type"

    async def test_critic_verdict_set_and_passed(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """``critic_verdict`` should be set by the placeholder critic (passed=True)."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        verdict = (
            result.get("critic_verdict")
            if isinstance(result, dict)
            else result.critic_verdict
        )
        assert verdict is not None, "critic_verdict should be set"
        assert verdict.passed is True, "Placeholder critic should pass"

    async def test_rag_search_called_with_query(
        self,
        graph,
        mock_rag: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """The RAG search should be called with the reformulated query."""
        await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        mock_rag.search.assert_awaited_once()
        call_args = mock_rag.search.call_args
        # First positional arg is the query
        query = call_args[0][0]
        assert "LoRA" in query, "RAG should search using the user's question"

    async def test_llm_invoked_for_synthesis(
        self,
        graph,
        mock_llm: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """The LLM should be invoked exactly once for answer synthesis."""
        await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        mock_llm.ainvoke.assert_awaited_once()

    async def test_retry_count_stays_zero(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """Retry count should remain 0 since the placeholder critic passes."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        retry_count = (
            result.get("retry_count", 0)
            if isinstance(result, dict)
            else result.retry_count
        )
        assert retry_count == 0, "No retries expected with passing critic"

    async def test_reasoning_trace_has_entries(
        self,
        graph,
        config: dict[str, Any],
    ) -> None:
        """Reasoning trace should contain entries from the strategy and formatter."""
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        trace = (
            result.get("reasoning_trace", [])
            if isinstance(result, dict)
            else result.reasoning_trace
        )
        assert len(trace) >= 2, "Should have trace entries from strategy + format_output"

    async def test_different_questions_produce_different_answers(
        self,
        mock_rag: MagicMock,
    ) -> None:
        """Different questions should flow through independently."""
        llm1 = _make_mock_llm("Answer about LoRA technique.")
        llm2 = _make_mock_llm("Answer about attention mechanism.")

        graph1 = build_main_graph(rag=mock_rag, llm=llm1)
        graph2 = build_main_graph(rag=mock_rag, llm=llm2)

        r1 = await graph1.ainvoke(
            {"question": "What is LoRA?"},
            config={"configurable": {"thread_id": "diff-q-1"}},
        )
        r2 = await graph2.ainvoke(
            {"question": "What is attention?"},
            config={"configurable": {"thread_id": "diff-q-2"}},
        )

        fa1 = r1.get("final_answer", "") if isinstance(r1, dict) else r1.final_answer
        fa2 = r2.get("final_answer", "") if isinstance(r2, dict) else r2.final_answer

        assert fa1 != fa2, "Different LLM responses should produce different final answers"
