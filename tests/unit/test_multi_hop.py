"""Unit tests for the Multi-hop (Plan-and-Execute) strategy node (C4).

Verifies that the multi-hop strategy correctly:
- Decomposes the question into sub-questions via LLM structured output.
- Executes sub-questions sequentially via RAG search.
- Supports replan: decision ``"next_step"`` continues, ``"replan"`` revises
  the plan, ``"synthesize"`` terminates the loop early.
- Terminates the plan→execute→replan loop (no infinite loop).
- Synthesizes a non-empty ``draft_answer`` from all retrieved contexts.
- Respects the ``max_plan_steps`` safety limit.
- Records reasoning trace steps (plan, action, observation, replan, synthesis).
- Raises ``ValueError`` when ``state.intent`` is ``None``.
- Factory creates a callable node with correct name and bound parameters.
- Placeholder returns a dict with ``draft_answer``.

All tests use mocked RAG and LLM — no real server connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.state import AgentState, Intent, ReasoningStep, RetrievedContext
from src.agent.strategies.multi_hop import (
    DEFAULT_MAX_PLAN_STEPS,
    DEFAULT_TOP_K,
    ReplanDecision,
    SubQuestionPlan,
    create_multi_hop_strategy_node,
    multi_hop_strategy_node,
    run_multi_hop_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    intent_type: str = "multi_hop",
    reformulated_query: str = "How did QLoRA build upon LoRA by introducing quantization?",
    confidence: float = 0.90,
    constraints: dict[str, Any] | None = None,
) -> Intent:
    """Create an Intent with defaults suitable for multi-hop testing."""
    return Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=reformulated_query,
        constraints=constraints or {"technique": "quantization"},
    )


def _make_state(
    *,
    question: str = "How did QLoRA build upon LoRA by introducing quantization?",
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
            content=f"{prefix}: QLoRA adds 4-bit NormalFloat quantization.",
            source=f"{prefix}_paper_qlora",
            doc_id=f"doc_{prefix}_2",
            relevance_score=0.88,
        ),
    ]


def _make_mock_rag(
    contexts_per_call: list[list[RetrievedContext]] | None = None,
) -> MagicMock:
    """Create a mock RAGToolWrapper with per-call context returns.

    By default returns 3 batches (one per sub-question in the default plan).
    """
    rag = MagicMock()
    if contexts_per_call is None:
        contexts_per_call = [
            _sample_contexts("s1"),
            _sample_contexts("s2"),
            _sample_contexts("s3"),
        ]
    rag.search = AsyncMock(side_effect=contexts_per_call)
    return rag


def _make_mock_llm(
    *,
    plan_questions: list[str] | None = None,
    replan_decisions: list[ReplanDecision] | None = None,
    revised_questions: list[str] | None = None,
    synthesis_text: str = (
        "QLoRA builds upon LoRA by adding 4-bit NormalFloat quantization "
        "to the base model weights while keeping LoRA adapters in higher precision."
    ),
) -> MagicMock:
    """Create a mock LLM supporting ``with_structured_output`` and ``ainvoke``.

    Args:
        plan_questions: Sub-questions returned by decompose (1st call) and
            optionally by replan (2nd call via *revised_questions*).
        replan_decisions: Sequence of ``ReplanDecision`` objects returned
            by successive ``_decide_next`` calls.
        revised_questions: If set, a second call to
            ``with_structured_output(SubQuestionPlan)`` returns this plan.
        synthesis_text: Text returned by the direct ``ainvoke`` for synthesis.
    """
    llm = MagicMock()

    if plan_questions is None:
        plan_questions = [
            "What is LoRA and how does it work?",
            "What is quantization in the context of LLM fine-tuning?",
            "How does QLoRA combine LoRA and quantization?",
        ]

    # -- Structured output mocks -------------------------------------------
    # Track calls to with_structured_output(SubQuestionPlan) so we can
    # differentiate initial decomposition from replanning.
    plan_call_counter = {"n": 0}
    all_plan_responses = [plan_questions]
    if revised_questions is not None:
        all_plan_responses.append(revised_questions)

    if replan_decisions is None:
        # Default: always continue (for a 3-question plan, decide is called
        # twice — after step 1 and step 2).
        replan_decisions = [
            ReplanDecision(action="next_step", reason="Continue as planned"),
            ReplanDecision(action="next_step", reason="Continue as planned"),
        ]
    replan_call_counter = {"n": 0}

    def _with_structured_output(output_cls):
        structured_mock = MagicMock()

        if output_cls is SubQuestionPlan:

            async def _plan_ainvoke(*args: Any, **kwargs: Any) -> SubQuestionPlan:
                idx = min(plan_call_counter["n"], len(all_plan_responses) - 1)
                plan_call_counter["n"] += 1
                return SubQuestionPlan(questions=all_plan_responses[idx])

            structured_mock.ainvoke = _plan_ainvoke

        elif output_cls is ReplanDecision:

            async def _decide_ainvoke(*args: Any, **kwargs: Any) -> ReplanDecision:
                idx = min(replan_call_counter["n"], len(replan_decisions) - 1)
                replan_call_counter["n"] += 1
                return replan_decisions[idx]

            structured_mock.ainvoke = _decide_ainvoke

        return structured_mock

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    # -- Direct ainvoke (synthesis) ----------------------------------------
    synthesis_response = MagicMock()
    synthesis_response.content = synthesis_text
    llm.ainvoke = AsyncMock(return_value=synthesis_response)

    return llm


# ---------------------------------------------------------------------------
# run_multi_hop_strategy — core logic
# ---------------------------------------------------------------------------


class TestRunMultiHopStrategy:
    """Tests for the core multi-hop strategy implementation."""

    async def test_basic_plan_execute_synthesize(self) -> None:
        """Happy path: plan 3 sub-questions, execute all, synthesize."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        # All 3 sub-questions executed
        assert mock_rag.search.await_count == 3
        assert result["draft_answer"]
        assert len(result["draft_answer"]) > 0

    async def test_sub_questions_populated(self) -> None:
        """sub_questions should be set from the LLM decomposition."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm(
            plan_questions=["Q1: What is LoRA?", "Q2: What is QLoRA?"],
            replan_decisions=[
                ReplanDecision(action="next_step", reason="ok"),
            ],
        )
        mock_rag.search = AsyncMock(side_effect=[
            _sample_contexts("a"),
            _sample_contexts("b"),
        ])

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["sub_questions"] == ["Q1: What is LoRA?", "Q2: What is QLoRA?"]

    async def test_retrieved_contexts_from_all_steps(self) -> None:
        """Contexts from all executed sub-questions should be collected."""
        state = _make_state()
        contexts_1 = _sample_contexts("s1")
        contexts_2 = _sample_contexts("s2")
        contexts_3 = _sample_contexts("s3")
        mock_rag = _make_mock_rag([contexts_1, contexts_2, contexts_3])
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert len(result["retrieved_contexts"]) == 6  # 2 per step * 3 steps
        assert result["retrieved_contexts"][:2] == contexts_1
        assert result["retrieved_contexts"][2:4] == contexts_2
        assert result["retrieved_contexts"][4:6] == contexts_3

    async def test_retrieval_queries_track_executed_steps(self) -> None:
        """retrieval_queries should contain all executed sub-questions."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        expected_qs = [
            "What is LoRA and how does it work?",
            "What is quantization in the context of LLM fine-tuning?",
            "How does QLoRA combine LoRA and quantization?",
        ]
        mock_llm = _make_mock_llm(plan_questions=expected_qs)

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["retrieval_queries"] == expected_qs

    async def test_rag_called_with_correct_top_k(self) -> None:
        """RAG search should respect the top_k parameter."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_multi_hop_strategy(state, mock_rag, mock_llm, top_k=7)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 7 or call[1].get("top_k") == 7

    async def test_synthesize_early_terminates_loop(self) -> None:
        """When replan decision is 'synthesize', loop ends and only 1 step executes."""
        state = _make_state()
        mock_rag = _make_mock_rag([_sample_contexts("s1")])
        mock_llm = _make_mock_llm(
            replan_decisions=[
                ReplanDecision(action="synthesize", reason="Enough information gathered"),
            ],
        )

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        # Only 1 step executed (then synthesize decision terminates loop)
        assert mock_rag.search.await_count == 1
        assert len(result["retrieval_queries"]) == 1
        assert result["draft_answer"]

    async def test_replan_branch_revises_plan(self) -> None:
        """When replan decision is 'replan', the plan is revised and loop continues."""
        revised = [
            "What is LoRA and how does it work?",     # already executed
            "What is NF4 quantization?",               # revised step
            "How does QLoRA achieve memory efficiency?",  # new step
        ]
        state = _make_state()
        mock_rag = _make_mock_rag([
            _sample_contexts("s1"),  # step 1 of original plan
            _sample_contexts("s2"),  # step 2 of revised plan
            _sample_contexts("s3"),  # step 3 of revised plan
        ])
        mock_llm = _make_mock_llm(
            replan_decisions=[
                ReplanDecision(action="replan", reason="Need to refocus on NF4"),
                ReplanDecision(action="next_step", reason="Continue with revised plan"),
            ],
            revised_questions=revised,
        )

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        # 3 total RAG calls: original step 1, then revised steps 2 and 3
        assert mock_rag.search.await_count == 3
        assert result["sub_questions"] == revised
        assert result["draft_answer"]

        # Check that replan trace exists
        replan_traces = [
            t for t in result["reasoning_trace"]
            if t.step_type == "thought" and "Revised plan" in t.content
        ]
        assert len(replan_traces) == 1

    async def test_max_plan_steps_limits_execution(self) -> None:
        """The max_plan_steps parameter should cap execution even with many sub-questions."""
        many_questions = [f"Sub-question {i}" for i in range(10)]
        state = _make_state()
        # Provide enough RAG results for 2 steps (max_plan_steps=2)
        mock_rag = _make_mock_rag([
            _sample_contexts("s1"),
            _sample_contexts("s2"),
        ])
        mock_llm = _make_mock_llm(
            plan_questions=many_questions,
            replan_decisions=[
                ReplanDecision(action="next_step", reason="ok"),
            ],
        )

        result = await run_multi_hop_strategy(
            state, mock_rag, mock_llm, max_plan_steps=2,
        )

        assert mock_rag.search.await_count == 2
        assert len(result["retrieval_queries"]) == 2

    async def test_draft_answer_non_empty(self) -> None:
        """Core acceptance criterion: draft_answer must be non-empty."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0

    async def test_reasoning_trace_recorded(self) -> None:
        """Trace should contain plan, action, observation, replan, and synthesis steps."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        trace = result["reasoning_trace"]
        assert all(isinstance(step, ReasoningStep) for step in trace)

        step_types = [s.step_type for s in trace]

        # Must have at least: plan (thought), action, observation, synthesis (thought)
        assert "thought" in step_types
        assert "action" in step_types
        assert "observation" in step_types

        # First step should be the plan
        assert trace[0].step_type == "thought"
        assert "Plan" in trace[0].content

        # Last step should be the synthesis
        assert trace[-1].step_type == "thought"
        assert "Synthesized" in trace[-1].content

    async def test_replan_decision_trace_present(self) -> None:
        """Replan decision steps should appear in the trace."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        decision_traces = [
            t for t in result["reasoning_trace"]
            if t.step_type == "thought" and "Replan decision" in t.content
        ]
        # For 3 sub-questions, _decide_next is called after step 1 and step 2
        assert len(decision_traces) == 2

    async def test_uses_reformulated_query(self) -> None:
        """Should use intent.reformulated_query for the decomposition."""
        intent = _make_intent(reformulated_query="Trace the evolution from LoRA to QLoRA")
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        # Verify LLM was called for decomposition (first with_structured_output call)
        assert mock_llm.with_structured_output.call_count >= 1
        assert result["draft_answer"]

    async def test_falls_back_to_question_when_reformulated_empty(self) -> None:
        """Should fall back to state.question when reformulated_query is empty."""
        intent = _make_intent(reformulated_query="")
        state = _make_state(question="How is QLoRA related to LoRA?", intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]

    async def test_raises_when_intent_is_none(self) -> None:
        """Should raise ValueError when state.intent is None."""
        state = AgentState(question="test")
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        with pytest.raises(ValueError, match="state.intent"):
            await run_multi_hop_strategy(state, mock_rag, mock_llm)

    async def test_empty_retrieval_still_synthesizes(self) -> None:
        """Even with empty RAG results, synthesis should still produce an answer."""
        state = _make_state()
        mock_rag = _make_mock_rag([[], [], []])
        mock_llm = _make_mock_llm(
            synthesis_text="Insufficient context to fully answer, but based on available info...",
        )

        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert result["retrieved_contexts"] == []

    async def test_plan_then_execute_then_replan_loop_terminates(self) -> None:
        """The plan -> execute -> replan loop must terminate (no infinite loop).

        This test verifies the core acceptance criterion: the loop terminates
        normally with a non-empty draft_answer.
        """
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        # If the loop were infinite, this would time out under pytest
        result = await run_multi_hop_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert len(result["retrieval_queries"]) > 0
        assert len(result["sub_questions"]) > 0


# ---------------------------------------------------------------------------
# create_multi_hop_strategy_node (factory)
# ---------------------------------------------------------------------------


class TestCreateMultiHopStrategyNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that accepts AgentState."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_multi_hop_strategy_node(mock_rag, mock_llm)
        state = _make_state()

        result = await node(state)

        assert "draft_answer" in result
        assert "sub_questions" in result
        assert "retrieved_contexts" in result
        assert "retrieval_queries" in result
        assert "reasoning_trace" in result

    async def test_factory_passes_custom_top_k(self) -> None:
        """Factory should respect the top_k parameter."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_multi_hop_strategy_node(mock_rag, mock_llm, top_k=10)
        state = _make_state()

        await node(state)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 10 or call[1].get("top_k") == 10

    async def test_factory_passes_custom_max_plan_steps(self) -> None:
        """Factory should respect the max_plan_steps parameter."""
        many_questions = [f"Q{i}" for i in range(10)]
        mock_rag = _make_mock_rag([_sample_contexts(f"s{i}") for i in range(2)])
        mock_llm = _make_mock_llm(
            plan_questions=many_questions,
            replan_decisions=[
                ReplanDecision(action="next_step", reason="ok"),
            ],
        )

        node = create_multi_hop_strategy_node(
            mock_rag, mock_llm, max_plan_steps=2,
        )
        state = _make_state()

        result = await node(state)

        assert len(result["retrieval_queries"]) == 2

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_multi_hop_strategy_node(mock_rag, mock_llm)

        assert node.__name__ == "multi_hop_strategy_node"


# ---------------------------------------------------------------------------
# multi_hop_strategy_node (default placeholder)
# ---------------------------------------------------------------------------


class TestMultiHopStrategyNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_draft_answer(self) -> None:
        """Placeholder should return a dict with draft_answer."""
        state = _make_state()
        result = multi_hop_strategy_node(state)

        assert "draft_answer" in result
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0
