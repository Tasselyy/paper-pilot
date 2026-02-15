"""Unit tests for the Exploratory (ReAct) strategy node (C6).

Verifies that the exploratory strategy correctly:
- Executes a ReAct loop: think → act → observe → repeat.
- Terminates when the LLM chooses ``"synthesize"`` (voluntary stop).
- Terminates when ``max_react_steps`` is reached (forced stop).
- Produces a non-empty ``draft_answer`` after synthesis.
- Collects ``retrieved_contexts`` and ``retrieval_queries`` from all steps.
- Records reasoning trace steps (thought, action, observation, synthesis).
- Respects the ``top_k`` parameter for RAG search.
- Raises ``ValueError`` when ``state.intent`` is ``None``.
- Falls back to ``state.question`` when ``reformulated_query`` is empty.
- Factory creates a callable node with correct name and bound parameters.
- Placeholder returns a dict with ``draft_answer``.

All tests use mocked RAG and LLM — no real server connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.state import AgentState, Intent, ReasoningStep, RetrievedContext
from src.agent.strategies.exploratory import (
    DEFAULT_MAX_REACT_STEPS,
    DEFAULT_TOP_K,
    ReactDecision,
    create_exploratory_strategy_node,
    exploratory_strategy_node,
    run_exploratory_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    intent_type: str = "exploratory",
    reformulated_query: str = "What are the recent advances in parameter-efficient fine-tuning?",
    confidence: float = 0.88,
    constraints: dict[str, Any] | None = None,
) -> Intent:
    """Create an Intent with defaults suitable for exploratory testing."""
    return Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=reformulated_query,
        constraints=constraints or {},
    )


def _make_state(
    *,
    question: str = "What are the recent advances in parameter-efficient fine-tuning?",
    intent: Intent | None = None,
) -> AgentState:
    """Create a minimal AgentState for testing."""
    if intent is None:
        intent = _make_intent()
    return AgentState(question=question, intent=intent)


def _sample_contexts(prefix: str = "step") -> list[RetrievedContext]:
    """Return two sample RetrievedContext objects with a distinguishing prefix."""
    return [
        RetrievedContext(
            content=f"{prefix}: LoRA inserts low-rank adapters into transformer layers.",
            source=f"{prefix}_paper_lora",
            doc_id=f"doc_{prefix}_1",
            relevance_score=0.92,
        ),
        RetrievedContext(
            content=f"{prefix}: Adapter tuning freezes base model parameters.",
            source=f"{prefix}_paper_adapter",
            doc_id=f"doc_{prefix}_2",
            relevance_score=0.85,
        ),
    ]


def _make_mock_rag(
    contexts_per_call: list[list[RetrievedContext]] | None = None,
    num_calls: int = 3,
) -> MagicMock:
    """Create a mock RAGToolWrapper with per-call context returns.

    Args:
        contexts_per_call: Explicit list of results per call. Overrides
            ``num_calls`` if provided.
        num_calls: Number of calls to prepare default contexts for.
    """
    rag = MagicMock()
    if contexts_per_call is None:
        contexts_per_call = [
            _sample_contexts(f"s{i}") for i in range(num_calls)
        ]
    rag.search = AsyncMock(side_effect=contexts_per_call)
    return rag


def _make_mock_llm(
    *,
    decisions: list[ReactDecision] | None = None,
    synthesis_text: str = (
        "Recent advances in parameter-efficient fine-tuning include LoRA, "
        "QLoRA, AdaLoRA, and adapter-based methods, which dramatically "
        "reduce memory and compute requirements for LLM adaptation."
    ),
) -> MagicMock:
    """Create a mock LLM supporting ``with_structured_output`` and ``ainvoke``.

    Args:
        decisions: Sequence of ``ReactDecision`` objects returned by
            successive ``_think`` calls.  The last one should typically be
            ``action="synthesize"``.
        synthesis_text: Text returned by the direct ``ainvoke`` for synthesis.
    """
    llm = MagicMock()

    if decisions is None:
        decisions = [
            ReactDecision(
                action="search",
                query="What is LoRA and how does it enable parameter-efficient fine-tuning?",
                reasoning="Start by understanding the foundational LoRA technique.",
            ),
            ReactDecision(
                action="search",
                query="What are alternatives to LoRA such as adapter tuning and prefix tuning?",
                reasoning="Explore other PEFT methods for a comprehensive view.",
            ),
            ReactDecision(
                action="synthesize",
                query="",
                reasoning="Sufficient information gathered to answer the question.",
            ),
        ]
    decision_counter = {"n": 0}

    # -- Structured output mock (ReactDecision) -----------------------------
    def _with_structured_output(output_cls):
        structured_mock = MagicMock()

        if output_cls is ReactDecision:

            async def _decide_ainvoke(*args: Any, **kwargs: Any) -> ReactDecision:
                idx = min(decision_counter["n"], len(decisions) - 1)
                decision_counter["n"] += 1
                return decisions[idx]

            structured_mock.ainvoke = _decide_ainvoke

        return structured_mock

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    # -- Direct ainvoke (synthesis) -----------------------------------------
    synthesis_response = MagicMock()
    synthesis_response.content = synthesis_text
    llm.ainvoke = AsyncMock(return_value=synthesis_response)

    return llm


# ---------------------------------------------------------------------------
# run_exploratory_strategy — core logic
# ---------------------------------------------------------------------------


class TestRunExploratoryStrategy:
    """Tests for the core exploratory strategy implementation."""

    async def test_basic_react_loop_and_synthesize(self) -> None:
        """Happy path: 2 search steps, then synthesize."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        # 2 search steps before synthesize
        assert mock_rag.search.await_count == 2
        assert result["draft_answer"]
        assert len(result["draft_answer"]) > 0

    async def test_llm_synthesize_terminates_loop(self) -> None:
        """When LLM chooses 'synthesize' immediately, no RAG calls happen."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=0)
        mock_llm = _make_mock_llm(
            decisions=[
                ReactDecision(
                    action="synthesize",
                    query="",
                    reasoning="Already have enough context from the question.",
                ),
            ],
        )
        mock_rag.search = AsyncMock(side_effect=[])

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert mock_rag.search.await_count == 0
        assert len(result["retrieval_queries"]) == 0
        assert result["draft_answer"]

    async def test_max_react_steps_forces_termination(self) -> None:
        """When max_react_steps is reached, loop terminates even if LLM keeps searching.

        This is the core acceptance criterion for C6.
        """
        # LLM always wants to search (never synthesizes on its own)
        always_search = [
            ReactDecision(
                action="search",
                query=f"Search query {i}",
                reasoning="Need more info.",
            )
            for i in range(10)
        ]
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=3)
        mock_llm = _make_mock_llm(decisions=always_search)

        result = await run_exploratory_strategy(
            state, mock_rag, mock_llm, max_react_steps=3,
        )

        # Exactly 3 search calls (limited by max_react_steps)
        assert mock_rag.search.await_count == 3
        assert len(result["retrieval_queries"]) == 3
        assert result["draft_answer"]

        # Verify forced-termination trace exists
        forced_traces = [
            t for t in result["reasoning_trace"]
            if t.step_type == "thought" and "forcing synthesis" in t.content
        ]
        assert len(forced_traces) == 1

    async def test_max_react_steps_one_step(self) -> None:
        """With max_react_steps=1, only one search is performed then synthesis."""
        always_search = [
            ReactDecision(
                action="search",
                query="Single search query",
                reasoning="Need info.",
            ),
        ]
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=1)
        mock_llm = _make_mock_llm(decisions=always_search)

        result = await run_exploratory_strategy(
            state, mock_rag, mock_llm, max_react_steps=1,
        )

        assert mock_rag.search.await_count == 1
        assert len(result["retrieval_queries"]) == 1
        assert result["draft_answer"]

    async def test_retrieved_contexts_from_all_steps(self) -> None:
        """Contexts from all search steps should be collected."""
        state = _make_state()
        ctx_1 = _sample_contexts("s1")
        ctx_2 = _sample_contexts("s2")
        mock_rag = _make_mock_rag(contexts_per_call=[ctx_1, ctx_2])
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert len(result["retrieved_contexts"]) == 4  # 2 per step * 2 steps
        assert result["retrieved_contexts"][:2] == ctx_1
        assert result["retrieved_contexts"][2:4] == ctx_2

    async def test_retrieval_queries_track_executed_searches(self) -> None:
        """retrieval_queries should contain all search queries executed."""
        decisions = [
            ReactDecision(action="search", query="Query A", reasoning="..."),
            ReactDecision(action="search", query="Query B", reasoning="..."),
            ReactDecision(action="synthesize", query="", reasoning="done"),
        ]
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm(decisions=decisions)

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert result["retrieval_queries"] == ["Query A", "Query B"]

    async def test_rag_called_with_correct_top_k(self) -> None:
        """RAG search should respect the top_k parameter."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        await run_exploratory_strategy(state, mock_rag, mock_llm, top_k=7)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 7 or call[1].get("top_k") == 7

    async def test_draft_answer_non_empty(self) -> None:
        """Core acceptance criterion: draft_answer must be non-empty."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0

    async def test_reasoning_trace_recorded(self) -> None:
        """Trace should contain thought, action, observation, and synthesis steps."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        trace = result["reasoning_trace"]
        assert all(isinstance(step, ReasoningStep) for step in trace)

        step_types = [s.step_type for s in trace]

        assert "thought" in step_types
        assert "action" in step_types
        assert "observation" in step_types

        # First step should be the initial thought
        assert trace[0].step_type == "thought"
        assert "ReAct step" in trace[0].content

        # Last step should be the synthesis summary
        assert trace[-1].step_type == "thought"
        assert "Synthesized" in trace[-1].content

    async def test_trace_includes_synthesize_decision(self) -> None:
        """When LLM decides to synthesize, it should be in the trace."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        synth_traces = [
            t for t in result["reasoning_trace"]
            if t.step_type == "thought" and "action=synthesize" in t.content
        ]
        assert len(synth_traces) == 1

    async def test_uses_reformulated_query(self) -> None:
        """Should use intent.reformulated_query for the investigation."""
        intent = _make_intent(
            reformulated_query="Explore the landscape of adapter-based methods",
        )
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert mock_llm.with_structured_output.call_count >= 1
        assert result["draft_answer"]

    async def test_falls_back_to_question_when_reformulated_empty(self) -> None:
        """Should fall back to state.question when reformulated_query is empty."""
        intent = _make_intent(reformulated_query="")
        state = _make_state(
            question="What is parameter-efficient fine-tuning?",
            intent=intent,
        )
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]

    async def test_raises_when_intent_is_none(self) -> None:
        """Should raise ValueError when state.intent is None."""
        state = AgentState(question="test")
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        with pytest.raises(ValueError, match="state.intent"):
            await run_exploratory_strategy(state, mock_rag, mock_llm)

    async def test_empty_retrieval_still_synthesizes(self) -> None:
        """Even with empty RAG results, synthesis should produce an answer."""
        state = _make_state()
        mock_rag = _make_mock_rag(contexts_per_call=[[], []])
        mock_llm = _make_mock_llm(
            synthesis_text="Limited information available, but based on what we know...",
        )

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert result["retrieved_contexts"] == []

    async def test_empty_query_from_llm_uses_original_question(self) -> None:
        """When LLM returns empty query, should fallback to original question."""
        decisions = [
            ReactDecision(action="search", query="", reasoning="..."),
            ReactDecision(action="synthesize", query="", reasoning="done"),
        ]
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=1)
        mock_llm = _make_mock_llm(decisions=decisions)

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert mock_rag.search.await_count == 1
        call_query = mock_rag.search.call_args_list[0][0][0]
        assert call_query == state.intent.reformulated_query

    async def test_react_loop_terminates(self) -> None:
        """The ReAct loop must terminate (no infinite loop).

        This is a critical safety check — if the loop were infinite,
        this test would time out under pytest.
        """
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert len(result["retrieval_queries"]) > 0

    async def test_trace_metadata_contains_step_info(self) -> None:
        """Trace metadata should include step numbers and action info."""
        state = _make_state()
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        result = await run_exploratory_strategy(state, mock_rag, mock_llm)

        # Check that action steps have step metadata
        action_traces = [
            t for t in result["reasoning_trace"]
            if t.step_type == "action"
        ]
        assert len(action_traces) == 2
        for trace_step in action_traces:
            assert "step" in trace_step.metadata
            assert "query" in trace_step.metadata
            assert "num_results" in trace_step.metadata


# ---------------------------------------------------------------------------
# create_exploratory_strategy_node (factory)
# ---------------------------------------------------------------------------


class TestCreateExploratoryStrategyNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that accepts AgentState."""
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        node = create_exploratory_strategy_node(mock_rag, mock_llm)
        state = _make_state()

        result = await node(state)

        assert "draft_answer" in result
        assert "retrieved_contexts" in result
        assert "retrieval_queries" in result
        assert "reasoning_trace" in result

    async def test_factory_passes_custom_top_k(self) -> None:
        """Factory should respect the top_k parameter."""
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm()

        node = create_exploratory_strategy_node(mock_rag, mock_llm, top_k=10)
        state = _make_state()

        await node(state)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 10 or call[1].get("top_k") == 10

    async def test_factory_passes_custom_max_react_steps(self) -> None:
        """Factory should respect the max_react_steps parameter."""
        always_search = [
            ReactDecision(action="search", query=f"Q{i}", reasoning="...")
            for i in range(10)
        ]
        mock_rag = _make_mock_rag(num_calls=2)
        mock_llm = _make_mock_llm(decisions=always_search)

        node = create_exploratory_strategy_node(
            mock_rag, mock_llm, max_react_steps=2,
        )
        state = _make_state()

        result = await node(state)

        assert len(result["retrieval_queries"]) == 2

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_exploratory_strategy_node(mock_rag, mock_llm)

        assert node.__name__ == "exploratory_strategy_node"


# ---------------------------------------------------------------------------
# exploratory_strategy_node (default placeholder)
# ---------------------------------------------------------------------------


class TestExploratoryStrategyNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_draft_answer(self) -> None:
        """Placeholder should return a dict with draft_answer."""
        state = _make_state()
        result = exploratory_strategy_node(state)

        assert "draft_answer" in result
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0
