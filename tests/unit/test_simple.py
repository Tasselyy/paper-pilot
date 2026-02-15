"""Unit tests for the Simple strategy node (B3).

Verifies that the simple strategy correctly:
- Uses ``intent.reformulated_query`` to search via RAG.
- Falls back to ``state.question`` when ``reformulated_query`` is empty.
- Passes retrieved contexts to the LLM with the proper prompt format.
- Writes ``draft_answer``, ``retrieved_contexts``, ``retrieval_queries``
  into the partial state update.
- Handles empty retrieval results gracefully (LLM still called).
- Raises ``ValueError`` when ``state.intent`` is ``None``.
- Records reasoning trace steps for observability.

All tests use mocked RAG and LLM — no real server connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, Intent, ReasoningStep, RetrievedContext
from src.agent.strategies.simple import (
    DEFAULT_TOP_K,
    create_simple_strategy_node,
    run_simple_strategy,
    simple_strategy_node,
)
from src.prompts.strategies import (
    SIMPLE_SYSTEM_PROMPT,
    format_simple_contexts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    intent_type: str = "factual",
    reformulated_query: str = "What is LoRA?",
    confidence: float = 0.95,
) -> Intent:
    """Create an Intent with sensible defaults for simple strategy testing."""
    return Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=reformulated_query,
    )


def _make_state(
    *,
    question: str = "What is LoRA?",
    intent: Intent | None = None,
) -> AgentState:
    """Create a minimal AgentState for testing."""
    if intent is None:
        intent = _make_intent()
    return AgentState(question=question, intent=intent)


def _sample_retrieved_contexts() -> list[RetrievedContext]:
    """Return sample RetrievedContext objects mimicking RAG output."""
    return [
        RetrievedContext(
            content="LoRA adds low-rank adapters to transformer weights.",
            source="LoRA: Low-Rank Adaptation",
            doc_id="doc_001",
            relevance_score=0.95,
            chunk_index=0,
        ),
        RetrievedContext(
            content="QLoRA combines quantization with LoRA for efficient fine-tuning.",
            source="QLoRA Paper",
            doc_id="doc_002",
            relevance_score=0.87,
            chunk_index=3,
        ),
    ]


def _make_mock_rag(
    search_results: list[RetrievedContext] | None = None,
) -> MagicMock:
    """Create a mock RAGToolWrapper with configurable search results."""
    rag = MagicMock()
    rag.search = AsyncMock(
        return_value=search_results if search_results is not None else _sample_retrieved_contexts()
    )
    return rag


def _make_mock_llm(response_text: str = "LoRA is a parameter-efficient fine-tuning method.") -> MagicMock:
    """Create a mock BaseChatModel that returns a fixed response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm


# ---------------------------------------------------------------------------
# format_simple_contexts (prompt helper)
# ---------------------------------------------------------------------------


class TestFormatSimpleContexts:
    """Tests for the context formatting helper."""

    def test_formats_multiple_contexts(self) -> None:
        """Multiple contexts should be numbered and separated."""
        contexts = [
            {"content": "First context.", "source": "Paper A"},
            {"content": "Second context.", "source": "Paper B"},
        ]
        result = format_simple_contexts(contexts)
        assert "[1] Source: Paper A" in result
        assert "First context." in result
        assert "[2] Source: Paper B" in result
        assert "Second context." in result

    def test_empty_contexts_returns_placeholder(self) -> None:
        """Empty context list should return a placeholder message."""
        result = format_simple_contexts([])
        assert "No relevant contexts" in result

    def test_missing_source_defaults_to_unknown(self) -> None:
        """Missing 'source' key should default to 'Unknown'."""
        contexts = [{"content": "Some text"}]
        result = format_simple_contexts(contexts)
        assert "Unknown" in result
        assert "Some text" in result

    def test_missing_content_defaults_to_empty(self) -> None:
        """Missing 'content' key should still format without error."""
        contexts = [{"source": "Paper X"}]
        result = format_simple_contexts(contexts)
        assert "[1] Source: Paper X" in result


# ---------------------------------------------------------------------------
# run_simple_strategy — core logic
# ---------------------------------------------------------------------------


class TestRunSimpleStrategy:
    """Tests for the core simple strategy implementation."""

    async def test_basic_retrieve_and_synthesize(self) -> None:
        """Should search RAG, invoke LLM, and return correct state update."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm("LoRA is a technique for efficient fine-tuning.")

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        # Verify RAG was called with reformulated_query
        mock_rag.search.assert_awaited_once_with(
            "What is LoRA?", top_k=DEFAULT_TOP_K
        )

        # Verify LLM was called
        mock_llm.ainvoke.assert_awaited_once()

        # Verify state update
        assert result["draft_answer"] == "LoRA is a technique for efficient fine-tuning."
        assert len(result["retrieved_contexts"]) == 2
        assert all(isinstance(ctx, RetrievedContext) for ctx in result["retrieved_contexts"])
        assert result["retrieval_queries"] == ["What is LoRA?"]

    async def test_draft_answer_is_non_empty(self) -> None:
        """draft_answer must be non-empty when LLM returns content."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm("A meaningful answer.")

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert len(result["draft_answer"]) > 0

    async def test_retrieved_contexts_populated(self) -> None:
        """retrieved_contexts should contain the RAG results."""
        contexts = _sample_retrieved_contexts()
        state = _make_state()
        mock_rag = _make_mock_rag(contexts)
        mock_llm = _make_mock_llm()

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        assert result["retrieved_contexts"] == contexts
        assert result["retrieved_contexts"][0].doc_id == "doc_001"
        assert result["retrieved_contexts"][1].source == "QLoRA Paper"

    async def test_uses_reformulated_query(self) -> None:
        """Should use intent.reformulated_query for RAG search."""
        intent = _make_intent(reformulated_query="Explain low-rank adaptation in detail")
        state = _make_state(question="what's lora?", intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_simple_strategy(state, mock_rag, mock_llm)

        mock_rag.search.assert_awaited_once_with(
            "Explain low-rank adaptation in detail", top_k=DEFAULT_TOP_K
        )

    async def test_falls_back_to_question_when_reformulated_empty(self) -> None:
        """Should fall back to state.question when reformulated_query is empty."""
        intent = _make_intent(reformulated_query="")
        state = _make_state(question="What is LoRA?", intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_simple_strategy(state, mock_rag, mock_llm)

        mock_rag.search.assert_awaited_once_with(
            "What is LoRA?", top_k=DEFAULT_TOP_K
        )

    async def test_llm_receives_system_and_user_messages(self) -> None:
        """LLM should receive system prompt + formatted user message."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_simple_strategy(state, mock_rag, mock_llm)

        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system + user

        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == SIMPLE_SYSTEM_PROMPT
        assert "What is LoRA?" in user_msg.content
        # Should contain formatted contexts
        assert "LoRA: Low-Rank Adaptation" in user_msg.content

    async def test_empty_retrieval_still_calls_llm(self) -> None:
        """LLM should still be invoked even with zero retrieval results."""
        state = _make_state()
        mock_rag = _make_mock_rag(search_results=[])
        mock_llm = _make_mock_llm("I don't have enough context to answer.")

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        mock_llm.ainvoke.assert_awaited_once()
        assert result["draft_answer"] == "I don't have enough context to answer."
        assert result["retrieved_contexts"] == []

    async def test_custom_top_k(self) -> None:
        """Should respect a custom top_k parameter."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_simple_strategy(state, mock_rag, mock_llm, top_k=10)

        mock_rag.search.assert_awaited_once_with("What is LoRA?", top_k=10)

    async def test_raises_when_intent_is_none(self) -> None:
        """Should raise ValueError when state.intent is None."""
        state = AgentState(question="test")
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        with pytest.raises(ValueError, match="state.intent"):
            await run_simple_strategy(state, mock_rag, mock_llm)

    async def test_reasoning_trace_recorded(self) -> None:
        """Should include reasoning trace steps for observability."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) >= 2
        assert all(isinstance(step, ReasoningStep) for step in trace)

        # First step: RAG search action
        assert trace[0].step_type == "action"
        assert "RAG search" in trace[0].content

        # Second step: synthesis thought
        assert trace[1].step_type == "thought"
        assert "Synthesized" in trace[1].content

    async def test_follow_up_intent_uses_simple_strategy(self) -> None:
        """follow_up intent type should work with the simple strategy."""
        intent = _make_intent(
            intent_type="follow_up",
            reformulated_query="Tell me more about LoRA rank selection",
        )
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm("LoRA rank is typically set to 4, 8, or 16.")

        result = await run_simple_strategy(state, mock_rag, mock_llm)

        mock_rag.search.assert_awaited_once_with(
            "Tell me more about LoRA rank selection", top_k=DEFAULT_TOP_K
        )
        assert result["draft_answer"]


# ---------------------------------------------------------------------------
# create_simple_strategy_node (factory)
# ---------------------------------------------------------------------------


class TestCreateSimpleStrategyNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that accepts AgentState."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_simple_strategy_node(mock_rag, mock_llm)
        state = _make_state()

        result = await node(state)

        assert "draft_answer" in result
        assert "retrieved_contexts" in result

    async def test_factory_passes_custom_top_k(self) -> None:
        """Factory should respect the top_k parameter."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_simple_strategy_node(mock_rag, mock_llm, top_k=3)
        state = _make_state()

        await node(state)

        mock_rag.search.assert_awaited_once_with("What is LoRA?", top_k=3)

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_simple_strategy_node(mock_rag, mock_llm)

        assert node.__name__ == "simple_strategy_node"


# ---------------------------------------------------------------------------
# simple_strategy_node (default placeholder)
# ---------------------------------------------------------------------------


class TestSimpleStrategyNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_draft_answer(self) -> None:
        """Placeholder should return a dict with draft_answer."""
        state = _make_state()
        result = simple_strategy_node(state)

        assert "draft_answer" in result
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0
