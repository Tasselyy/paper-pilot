"""Unit tests for the retry_refine node (D3).

Verifies that the retry_refine node correctly:
- Uses critic feedback to rewrite state.question via Cloud LLM.
- Produces a refined question that contains feedback keywords.
- Increments retry_count by 1.
- Clears draft_answer for the next strategy cycle.
- Records a reasoning trace step of type 'critique'.
- Works through the ``create_retry_refine_node`` factory.
- Falls back to the synchronous ``retry_refine_node`` placeholder.
- Handles missing/empty feedback gracefully.
- Accepts both dict and AgentState state forms.

All tests use a mocked LLM — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.retry_refine import (
    RefineOutput,
    create_retry_refine_node,
    retry_refine_node,
    run_retry_refine,
)
from src.agent.state import (
    AgentState,
    CriticVerdict,
    Intent,
    ReasoningStep,
    RetrievedContext,
)
from src.prompts.retry_refine import (
    RETRY_REFINE_SYSTEM_PROMPT,
    RETRY_REFINE_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    question: str = "What is LoRA?",
    draft_answer: str = "LoRA is a parameter-efficient fine-tuning method.",
    intent_type: str = "factual",
    retry_count: int = 0,
    feedback: str = "Answer misses details about rank decomposition and memory savings.",
    critic_passed: bool = False,
    critic_score: float = 4.5,
) -> AgentState:
    """Create a minimal AgentState for testing retry_refine."""
    return AgentState(
        question=question,
        draft_answer=draft_answer,
        intent=Intent(
            type=intent_type,
            confidence=0.9,
            entities=["LoRA"],
            reformulated_query=question,
        ),
        critic_verdict=CriticVerdict(
            passed=critic_passed,
            score=critic_score,
            completeness=0.4,
            faithfulness=0.5,
            feedback=feedback,
        ),
        retry_count=retry_count,
        retrieved_contexts=[
            RetrievedContext(
                content="LoRA adds low-rank matrices to transformer layers.",
                source="LoRA Paper (2021)",
                doc_id="lora-001",
                relevance_score=0.92,
            ),
        ],
    )


def _make_mock_llm(
    *,
    refined_question: str = "What is LoRA, specifically its rank decomposition mechanism and memory savings compared to full fine-tuning?",
    refinement_summary: str = "Added emphasis on rank decomposition and memory savings as suggested by critic feedback.",
) -> MagicMock:
    """Create a mock LLM that returns a ``RefineOutput`` via structured output.

    Simulates ``llm.with_structured_output(RefineOutput).ainvoke(...)``
    returning a ``RefineOutput`` Pydantic instance.
    """
    refine_output = RefineOutput(
        refined_question=refined_question,
        refinement_summary=refinement_summary,
    )

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=refine_output)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    return llm


# ---------------------------------------------------------------------------
# RefineOutput model
# ---------------------------------------------------------------------------


class TestRefineOutput:
    """Tests for the RefineOutput Pydantic model."""

    def test_valid_output(self) -> None:
        """Should accept valid refined_question and refinement_summary."""
        output = RefineOutput(
            refined_question="What is LoRA and how does rank decomposition work?",
            refinement_summary="Added focus on rank decomposition.",
        )
        assert "rank decomposition" in output.refined_question
        assert len(output.refinement_summary) > 0

    def test_empty_strings_accepted(self) -> None:
        """Should accept empty strings (edge case, LLM may produce these)."""
        output = RefineOutput(
            refined_question="",
            refinement_summary="",
        )
        assert output.refined_question == ""
        assert output.refinement_summary == ""


# ---------------------------------------------------------------------------
# run_retry_refine — question refinement
# ---------------------------------------------------------------------------


class TestRunRetryRefineQuestion:
    """Tests verifying question refinement via LLM."""

    async def test_question_updated_with_feedback_keywords(self) -> None:
        """Refined question should contain keywords from the critic feedback."""
        state = _make_state(
            feedback="Answer misses details about rank decomposition and memory savings.",
        )
        mock_llm = _make_mock_llm(
            refined_question=(
                "What is LoRA, specifically its rank decomposition "
                "mechanism and memory savings compared to full fine-tuning?"
            ),
        )

        result = await run_retry_refine(state, mock_llm)

        assert "rank decomposition" in result["question"]
        assert "memory" in result["question"].lower()

    async def test_question_differs_from_original(self) -> None:
        """Refined question should differ from the original."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm(
            refined_question="What is LoRA and how does low-rank adaptation reduce parameters?",
        )

        result = await run_retry_refine(state, mock_llm)

        assert result["question"] != "What is LoRA?"

    async def test_question_preserves_topic(self) -> None:
        """Refined question should still be about the original topic."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm(
            refined_question="What is LoRA, including its rank parameters and training efficiency?",
        )

        result = await run_retry_refine(state, mock_llm)

        assert "LoRA" in result["question"]


# ---------------------------------------------------------------------------
# run_retry_refine — retry_count
# ---------------------------------------------------------------------------


class TestRunRetryRefineCount:
    """Tests verifying retry_count incrementation."""

    async def test_retry_count_incremented_from_zero(self) -> None:
        """retry_count should go from 0 to 1."""
        state = _make_state(retry_count=0)
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["retry_count"] == 1

    async def test_retry_count_incremented_from_one(self) -> None:
        """retry_count should go from 1 to 2."""
        state = _make_state(retry_count=1)
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["retry_count"] == 2

    async def test_retry_count_always_increments_by_one(self) -> None:
        """retry_count should always increment by exactly 1."""
        for initial in [0, 1, 2, 5]:
            state = _make_state(retry_count=initial)
            mock_llm = _make_mock_llm()

            result = await run_retry_refine(state, mock_llm)

            assert result["retry_count"] == initial + 1


# ---------------------------------------------------------------------------
# run_retry_refine — draft_answer cleared
# ---------------------------------------------------------------------------


class TestRunRetryRefineClearsDraft:
    """Tests verifying draft_answer is cleared for the next cycle."""

    async def test_draft_answer_cleared(self) -> None:
        """draft_answer should be empty after retry_refine."""
        state = _make_state(draft_answer="This is a stale draft.")
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["draft_answer"] == ""

    async def test_draft_answer_cleared_even_when_already_empty(self) -> None:
        """draft_answer should be empty even if it was already empty."""
        state = _make_state(draft_answer="")
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["draft_answer"] == ""


# ---------------------------------------------------------------------------
# run_retry_refine — reasoning trace
# ---------------------------------------------------------------------------


class TestRunRetryRefineTrace:
    """Tests for reasoning trace output from retry_refine."""

    async def test_trace_step_recorded(self) -> None:
        """Should include a reasoning trace step of type 'critique'."""
        state = _make_state()
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) == 1
        step = trace[0]
        assert isinstance(step, ReasoningStep)
        assert step.step_type == "critique"

    async def test_trace_contains_retry_info(self) -> None:
        """Reasoning trace content should mention the retry count."""
        state = _make_state(retry_count=0)
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        step = result["reasoning_trace"][0]
        assert "retry" in step.content.lower() or "Retry" in step.content

    async def test_trace_metadata_contains_questions(self) -> None:
        """Reasoning trace metadata should contain both original and refined questions."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm(
            refined_question="What is LoRA and its rank decomposition?",
        )

        result = await run_retry_refine(state, mock_llm)

        metadata = result["reasoning_trace"][0].metadata
        assert metadata["original_question"] == "What is LoRA?"
        assert metadata["refined_question"] == "What is LoRA and its rank decomposition?"
        assert "retry_count" in metadata
        assert metadata["retry_count"] == 1


# ---------------------------------------------------------------------------
# run_retry_refine — LLM interaction
# ---------------------------------------------------------------------------


class TestRunRetryRefineLLMInteraction:
    """Tests verifying correct LLM prompt construction."""

    async def test_llm_receives_structured_output_schema(self) -> None:
        """LLM should receive RefineOutput as the structured output schema."""
        state = _make_state()
        mock_llm = _make_mock_llm()

        await run_retry_refine(state, mock_llm)

        mock_llm.with_structured_output.assert_called_once_with(RefineOutput)

    async def test_llm_receives_correct_messages(self) -> None:
        """LLM should receive system prompt and formatted user message."""
        state = _make_state(
            question="What is LoRA?",
            draft_answer="LoRA is a fine-tuning method.",
            feedback="Missing details about rank parameters.",
        )
        mock_llm = _make_mock_llm()

        await run_retry_refine(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]

        assert len(call_args) == 2
        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == RETRY_REFINE_SYSTEM_PROMPT
        assert "What is LoRA?" in user_msg.content
        assert "LoRA is a fine-tuning method." in user_msg.content
        assert "Missing details about rank parameters." in user_msg.content

    async def test_prompt_includes_strategy(self) -> None:
        """User prompt should include the strategy name."""
        state = _make_state(intent_type="comparative")
        mock_llm = _make_mock_llm()

        await run_retry_refine(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]
        user_msg = call_args[1]

        assert "comparative" in user_msg.content

    async def test_prompt_includes_feedback(self) -> None:
        """User prompt should include the critic's feedback."""
        feedback = "The answer lacks discussion of memory efficiency."
        state = _make_state(feedback=feedback)
        mock_llm = _make_mock_llm()

        await run_retry_refine(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]
        user_msg = call_args[1]

        assert feedback in user_msg.content


# ---------------------------------------------------------------------------
# run_retry_refine — empty/missing feedback
# ---------------------------------------------------------------------------


class TestRunRetryRefineNoFeedback:
    """Tests for edge-case: empty or missing critic feedback."""

    async def test_empty_feedback_falls_back_to_generic(self) -> None:
        """Empty feedback should produce a generic refinement without LLM call."""
        state = _make_state(feedback="")
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert "comprehensive" in result["question"].lower() or "detailed" in result["question"].lower()
        mock_llm.with_structured_output.assert_not_called()

    async def test_whitespace_only_feedback_falls_back(self) -> None:
        """Whitespace-only feedback should also fall back to generic."""
        state = _make_state(feedback="   \n\t  ")
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["question"] != "What is LoRA?"
        mock_llm.with_structured_output.assert_not_called()

    async def test_no_critic_verdict_falls_back(self) -> None:
        """Missing critic_verdict should fall back to generic refinement."""
        state = AgentState(
            question="What is LoRA?",
            draft_answer="Some draft.",
            retry_count=0,
        )
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["retry_count"] == 1
        assert result["question"] != "What is LoRA?"
        mock_llm.with_structured_output.assert_not_called()

    async def test_retry_count_still_increments_without_feedback(self) -> None:
        """retry_count should increment even without feedback."""
        state = _make_state(feedback="", retry_count=1)
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["retry_count"] == 2


# ---------------------------------------------------------------------------
# run_retry_refine — dict state access
# ---------------------------------------------------------------------------


class TestRunRetryRefineDictState:
    """Tests for dict-like state access (LangGraph compatibility)."""

    async def test_dict_state_works(self) -> None:
        """Should work when state is a plain dict."""
        state = {
            "question": "What is LoRA?",
            "draft_answer": "LoRA is a fine-tuning method.",
            "retry_count": 0,
            "intent": Intent(
                type="factual",
                confidence=0.9,
                reformulated_query="What is LoRA?",
            ),
            "critic_verdict": CriticVerdict(
                passed=False,
                score=4.0,
                completeness=0.4,
                faithfulness=0.5,
                feedback="Needs more detail on rank decomposition.",
            ),
        }
        mock_llm = _make_mock_llm(
            refined_question="What is LoRA and how does rank decomposition work?",
        )

        result = await run_retry_refine(state, mock_llm)

        assert "rank decomposition" in result["question"]
        assert result["retry_count"] == 1
        assert result["draft_answer"] == ""

    async def test_dict_state_missing_intent(self) -> None:
        """Should handle missing intent in dict state (strategy defaults to unknown)."""
        state = {
            "question": "What is LoRA?",
            "draft_answer": "LoRA is a fine-tuning method.",
            "retry_count": 0,
            "critic_verdict": CriticVerdict(
                passed=False,
                score=4.0,
                completeness=0.4,
                faithfulness=0.5,
                feedback="Needs more detail.",
            ),
        }
        mock_llm = _make_mock_llm()

        result = await run_retry_refine(state, mock_llm)

        assert result["retry_count"] == 1


# ---------------------------------------------------------------------------
# create_retry_refine_node (factory)
# ---------------------------------------------------------------------------


class TestCreateRetryRefineNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that processes state."""
        mock_llm = _make_mock_llm()

        node = create_retry_refine_node(mock_llm)
        state = _make_state()

        result = await node(state)

        assert "question" in result
        assert result["retry_count"] == 1

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_llm = _make_mock_llm()

        node = create_retry_refine_node(mock_llm)

        assert node.__name__ == "retry_refine_node"

    async def test_factory_delegates_to_run_retry_refine(self) -> None:
        """Factory node should produce same structure as run_retry_refine."""
        mock_llm = _make_mock_llm(
            refined_question="What is LoRA and its rank decomposition?",
            refinement_summary="Added rank decomposition focus.",
        )

        node = create_retry_refine_node(mock_llm)
        state = _make_state()

        result = await node(state)

        assert "question" in result
        assert "retry_count" in result
        assert "draft_answer" in result
        assert "reasoning_trace" in result
        assert result["draft_answer"] == ""
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# retry_refine_node (default sync placeholder)
# ---------------------------------------------------------------------------


class TestRetryRefineNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_increments_retry_count(self) -> None:
        """Placeholder should increment retry_count."""
        state = _make_state(retry_count=0)
        result = retry_refine_node(state)

        assert result["retry_count"] == 1

    def test_placeholder_updates_question(self) -> None:
        """Placeholder should update the question with feedback."""
        state = _make_state(
            question="What is LoRA?",
            feedback="Missing rank decomposition details.",
        )
        result = retry_refine_node(state)

        assert result["question"] != "What is LoRA?"
        assert "rank decomposition" in result["question"].lower()

    def test_placeholder_clears_draft_answer(self) -> None:
        """Placeholder should clear draft_answer."""
        state = _make_state(draft_answer="Old draft.")
        result = retry_refine_node(state)

        assert result["draft_answer"] == ""

    def test_placeholder_includes_trace(self) -> None:
        """Placeholder should include a reasoning trace step."""
        state = _make_state()
        result = retry_refine_node(state)

        assert "reasoning_trace" in result
        assert len(result["reasoning_trace"]) == 1
        step = result["reasoning_trace"][0]
        assert isinstance(step, ReasoningStep)
        assert step.step_type == "critique"

    def test_placeholder_handles_no_feedback(self) -> None:
        """Placeholder should handle empty feedback gracefully."""
        state = AgentState(
            question="What is LoRA?",
            draft_answer="Some draft.",
            retry_count=0,
        )
        result = retry_refine_node(state)

        assert result["retry_count"] == 1
        assert result["question"] != "What is LoRA?"
        assert "comprehensive" in result["question"].lower() or "detailed" in result["question"].lower()

    def test_placeholder_works_with_dict_state(self) -> None:
        """Placeholder should work with dict state."""
        state = {
            "question": "What is LoRA?",
            "retry_count": 1,
            "critic_verdict": CriticVerdict(
                passed=False,
                score=4.0,
                completeness=0.4,
                faithfulness=0.5,
                feedback="Needs improvement.",
            ),
        }
        result = retry_refine_node(state)

        assert result["retry_count"] == 2
        assert "improvement" in result["question"].lower()
