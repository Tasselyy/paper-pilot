"""Unit tests for the Critic node (D1).

Verifies that the Critic correctly:
- Evaluates a draft answer and produces a CriticVerdict.
- Sets passed=True when score >= 7.0 and passed=False when score < 7.0.
- Includes non-empty feedback in both pass and fail cases.
- Records a reasoning trace step of type 'critique'.
- Handles empty draft_answer gracefully (auto-fail with score=0).
- Works through the ``create_critic_node`` factory.
- Falls back to the synchronous ``critic_node`` placeholder.
- Accepts both dict and AgentState state forms.

All tests use a mocked LLM — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.critic import (
    CRITIC_PASS_THRESHOLD,
    CriticOutput,
    create_critic_node,
    critic_node,
    run_critic,
)
from src.agent.state import (
    AgentState,
    CriticVerdict,
    Intent,
    ReasoningStep,
    RetrievedContext,
)
from src.prompts.critic import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    question: str = "What is LoRA?",
    draft_answer: str = "LoRA is a parameter-efficient fine-tuning method.",
    intent_type: str = "factual",
    with_contexts: bool = True,
) -> AgentState:
    """Create a minimal AgentState for testing the Critic."""
    contexts = []
    if with_contexts:
        contexts = [
            RetrievedContext(
                content="LoRA adds low-rank matrices to transformer layers.",
                source="LoRA Paper (2021)",
                doc_id="lora-001",
                relevance_score=0.92,
            ),
            RetrievedContext(
                content="Low-Rank Adaptation reduces trainable params by 10000x.",
                source="LoRA Paper (2021)",
                doc_id="lora-001",
                relevance_score=0.88,
            ),
        ]
    return AgentState(
        question=question,
        draft_answer=draft_answer,
        intent=Intent(
            type=intent_type,
            confidence=0.9,
            entities=["LoRA"],
            reformulated_query=question,
        ),
        retrieved_contexts=contexts,
    )


def _make_mock_llm(
    *,
    score: float = 8.5,
    completeness: float = 0.85,
    faithfulness: float = 0.90,
    feedback: str = "Good answer, well-grounded in the sources.",
) -> MagicMock:
    """Create a mock LLM that returns a ``CriticOutput`` via structured output.

    Simulates ``llm.with_structured_output(CriticOutput).ainvoke(...)``
    returning a ``CriticOutput`` Pydantic instance.
    """
    critic_output = CriticOutput(
        score=score,
        completeness=completeness,
        faithfulness=faithfulness,
        feedback=feedback,
    )

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=critic_output)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    return llm


class _MockLocalCritic:
    """Simple local critic stub for fallback tests."""

    def __init__(
        self,
        *,
        score: float = 8.0,
        completeness: float = 0.8,
        faithfulness: float = 0.85,
        feedback: str = "Good local evaluation.",
        should_raise: bool = False,
    ) -> None:
        self.score = score
        self.completeness = completeness
        self.faithfulness = faithfulness
        self.feedback = feedback
        self.should_raise = should_raise
        self.calls: list[dict[str, object]] = []

    def evaluate_answer(
        self,
        *,
        question: str,
        draft_answer: str,
        retrieved_contexts: list[object] | None = None,
        strategy: str = "unknown",
        pass_threshold: float = 7.0,
    ) -> dict[str, float | bool | str]:
        """Return deterministic local critic metrics or raise."""
        self.calls.append(
            {
                "question": question,
                "draft_answer": draft_answer,
                "retrieved_contexts": retrieved_contexts or [],
                "strategy": strategy,
                "pass_threshold": pass_threshold,
            }
        )
        if self.should_raise:
            raise RuntimeError("local critic unavailable")
        return {
            "passed": self.score >= pass_threshold,
            "score": self.score,
            "completeness": self.completeness,
            "faithfulness": self.faithfulness,
            "feedback": self.feedback,
        }


# ---------------------------------------------------------------------------
# CriticOutput model
# ---------------------------------------------------------------------------


class TestCriticOutput:
    """Tests for the CriticOutput Pydantic model."""

    def test_valid_output(self) -> None:
        """Should accept valid score, completeness, faithfulness, and feedback."""
        output = CriticOutput(
            score=8.0,
            completeness=0.9,
            faithfulness=0.85,
            feedback="Great answer.",
        )
        assert output.score == 8.0
        assert output.completeness == 0.9
        assert output.faithfulness == 0.85
        assert output.feedback == "Great answer."

    def test_score_bounds(self) -> None:
        """Score must be in [0, 10]."""
        CriticOutput(score=0.0, completeness=0.5, faithfulness=0.5, feedback="f")
        CriticOutput(score=10.0, completeness=0.5, faithfulness=0.5, feedback="f")

        with pytest.raises(Exception):
            CriticOutput(score=-0.1, completeness=0.5, faithfulness=0.5, feedback="f")
        with pytest.raises(Exception):
            CriticOutput(score=10.1, completeness=0.5, faithfulness=0.5, feedback="f")

    def test_completeness_bounds(self) -> None:
        """Completeness must be in [0, 1]."""
        CriticOutput(score=5.0, completeness=0.0, faithfulness=0.5, feedback="f")
        CriticOutput(score=5.0, completeness=1.0, faithfulness=0.5, feedback="f")

        with pytest.raises(Exception):
            CriticOutput(score=5.0, completeness=-0.1, faithfulness=0.5, feedback="f")
        with pytest.raises(Exception):
            CriticOutput(score=5.0, completeness=1.1, faithfulness=0.5, feedback="f")

    def test_faithfulness_bounds(self) -> None:
        """Faithfulness must be in [0, 1]."""
        CriticOutput(score=5.0, completeness=0.5, faithfulness=0.0, feedback="f")
        CriticOutput(score=5.0, completeness=0.5, faithfulness=1.0, feedback="f")

        with pytest.raises(Exception):
            CriticOutput(score=5.0, completeness=0.5, faithfulness=-0.1, feedback="f")
        with pytest.raises(Exception):
            CriticOutput(score=5.0, completeness=0.5, faithfulness=1.1, feedback="f")


# ---------------------------------------------------------------------------
# run_critic — pass scenario
# ---------------------------------------------------------------------------


class TestRunCriticPass:
    """Tests for the Critic when the answer passes (score >= 7.0)."""

    async def test_high_score_passes(self) -> None:
        """An answer scoring >= 7.0 should have passed=True."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=8.5, completeness=0.85, faithfulness=0.90)

        result = await run_critic(state, mock_llm)

        verdict = result["critic_verdict"]
        assert isinstance(verdict, CriticVerdict)
        assert verdict.passed is True
        assert verdict.score == 8.5
        assert verdict.completeness == 0.85
        assert verdict.faithfulness == 0.90

    async def test_score_exactly_at_threshold_passes(self) -> None:
        """An answer scoring exactly 7.0 should pass."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=7.0)

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is True
        assert result["critic_verdict"].score >= CRITIC_PASS_THRESHOLD

    async def test_passed_has_feedback(self) -> None:
        """Even when passed, feedback should be non-empty."""
        state = _make_state()
        mock_llm = _make_mock_llm(
            score=9.0,
            feedback="Excellent, comprehensive answer.",
        )

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is True
        assert len(result["critic_verdict"].feedback) > 0


# ---------------------------------------------------------------------------
# run_critic — fail scenario
# ---------------------------------------------------------------------------


class TestRunCriticFail:
    """Tests for the Critic when the answer fails (score < 7.0)."""

    async def test_low_score_fails(self) -> None:
        """An answer scoring < 7.0 should have passed=False."""
        state = _make_state()
        mock_llm = _make_mock_llm(
            score=4.5,
            completeness=0.40,
            faithfulness=0.50,
            feedback="Answer misses key details about rank decomposition.",
        )

        result = await run_critic(state, mock_llm)

        verdict = result["critic_verdict"]
        assert isinstance(verdict, CriticVerdict)
        assert verdict.passed is False
        assert verdict.score == 4.5
        assert verdict.score < CRITIC_PASS_THRESHOLD

    async def test_failed_has_nonempty_feedback(self) -> None:
        """When failed, feedback must be non-empty and actionable."""
        state = _make_state()
        mock_llm = _make_mock_llm(
            score=3.0,
            feedback="Missing discussion of rank parameters and training details.",
        )

        result = await run_critic(state, mock_llm)

        verdict = result["critic_verdict"]
        assert verdict.passed is False
        assert len(verdict.feedback) > 0
        assert "missing" in verdict.feedback.lower() or len(verdict.feedback) > 5

    async def test_score_just_below_threshold_fails(self) -> None:
        """An answer scoring just below 7.0 should fail."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=6.9)

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is False


# ---------------------------------------------------------------------------
# run_critic — reasoning trace
# ---------------------------------------------------------------------------


class TestRunCriticTrace:
    """Tests for reasoning trace output from the Critic."""

    async def test_trace_step_recorded(self) -> None:
        """Should include a reasoning trace step of type 'critique'."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=8.0)

        result = await run_critic(state, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) == 1
        step = trace[0]
        assert isinstance(step, ReasoningStep)
        assert step.step_type == "critique"

    async def test_trace_contains_score_info(self) -> None:
        """Reasoning trace content should mention the score and passed status."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=8.5, completeness=0.85, faithfulness=0.90)

        result = await run_critic(state, mock_llm)

        step = result["reasoning_trace"][0]
        assert "8.5" in step.content
        assert "True" in step.content or "passed=True" in step.content

    async def test_trace_metadata_fields(self) -> None:
        """Reasoning trace metadata should contain score, passed, feedback_preview."""
        state = _make_state()
        mock_llm = _make_mock_llm(
            score=5.0,
            completeness=0.4,
            faithfulness=0.5,
            feedback="Needs more detail on rank decomposition.",
        )

        result = await run_critic(state, mock_llm)

        metadata = result["reasoning_trace"][0].metadata
        assert metadata["score"] == 5.0
        assert metadata["completeness"] == 0.4
        assert metadata["faithfulness"] == 0.5
        assert metadata["passed"] is False
        assert "feedback_preview" in metadata


# ---------------------------------------------------------------------------
# run_critic — empty draft_answer
# ---------------------------------------------------------------------------


class TestRunCriticEmptyDraft:
    """Tests for edge-case: empty draft answer."""

    async def test_empty_draft_auto_fails(self) -> None:
        """Empty draft_answer should auto-fail with score=0."""
        state = _make_state(draft_answer="")
        mock_llm = _make_mock_llm()

        result = await run_critic(state, mock_llm)

        verdict = result["critic_verdict"]
        assert verdict.passed is False
        assert verdict.score == 0.0
        assert "empty" in verdict.feedback.lower()

    async def test_empty_draft_skips_llm(self) -> None:
        """LLM should NOT be called when draft is empty."""
        state = _make_state(draft_answer="")
        mock_llm = _make_mock_llm()

        await run_critic(state, mock_llm)

        mock_llm.with_structured_output.assert_not_called()

    async def test_whitespace_only_draft_auto_fails(self) -> None:
        """Whitespace-only draft_answer should also auto-fail."""
        state = _make_state(draft_answer="   \n\t  ")
        mock_llm = _make_mock_llm()

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is False
        assert result["critic_verdict"].score == 0.0


# ---------------------------------------------------------------------------
# run_critic — LLM interaction
# ---------------------------------------------------------------------------


class TestRunCriticLLMInteraction:
    """Tests verifying correct LLM prompt construction."""

    async def test_llm_receives_structured_output_schema(self) -> None:
        """LLM should receive CriticOutput as the structured output schema."""
        state = _make_state()
        mock_llm = _make_mock_llm()

        await run_critic(state, mock_llm)

        mock_llm.with_structured_output.assert_called_once_with(CriticOutput)

    async def test_llm_receives_correct_messages(self) -> None:
        """LLM should receive system prompt and formatted user message."""
        state = _make_state(
            question="What is LoRA?",
            draft_answer="LoRA is a fine-tuning method.",
        )
        mock_llm = _make_mock_llm()

        await run_critic(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]

        assert len(call_args) == 2
        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == CRITIC_SYSTEM_PROMPT
        assert "What is LoRA?" in user_msg.content
        assert "LoRA is a fine-tuning method." in user_msg.content

    async def test_prompt_includes_strategy(self) -> None:
        """User prompt should include the strategy name."""
        state = _make_state(intent_type="comparative")
        mock_llm = _make_mock_llm()

        await run_critic(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]
        user_msg = call_args[1]

        assert "comparative" in user_msg.content

    async def test_prompt_includes_contexts(self) -> None:
        """User prompt should include formatted retrieved contexts."""
        state = _make_state(with_contexts=True)
        mock_llm = _make_mock_llm()

        await run_critic(state, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]
        user_msg = call_args[1]

        assert "LoRA Paper (2021)" in user_msg.content
        assert "low-rank matrices" in user_msg.content


# ---------------------------------------------------------------------------
# run_critic — dict state access
# ---------------------------------------------------------------------------


class TestRunCriticDictState:
    """Tests for dict-like state access (LangGraph compatibility)."""

    async def test_dict_state_works(self) -> None:
        """Should work when state is a plain dict."""
        state = {
            "question": "What is LoRA?",
            "draft_answer": "LoRA is a fine-tuning method.",
            "retrieved_contexts": [],
            "intent": Intent(
                type="factual",
                confidence=0.9,
                reformulated_query="What is LoRA?",
            ),
        }
        mock_llm = _make_mock_llm(score=8.0)

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is True

    async def test_fallback_local_critic_preferred_when_available(self) -> None:
        """Local critic result should be used when local inference succeeds."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=4.0, feedback="Cloud should not be used.")
        local_critic = _MockLocalCritic(
            score=8.8,
            completeness=0.86,
            faithfulness=0.91,
            feedback="Local critic verdict.",
        )

        result = await run_critic(state, mock_llm, local_critic=local_critic)

        verdict = result["critic_verdict"]
        assert verdict.passed is True
        assert verdict.score == pytest.approx(8.8, abs=1e-6)
        assert verdict.feedback == "Local critic verdict."
        assert len(local_critic.calls) == 1
        mock_llm.with_structured_output.assert_not_called()
        assert result["reasoning_trace"][0].metadata["source"] == "local"

    async def test_fallback_to_cloud_when_local_critic_fails(self) -> None:
        """Critic should fall back to cloud evaluation on local errors."""
        state = _make_state()
        mock_llm = _make_mock_llm(score=7.6, feedback="Cloud fallback verdict.")
        local_critic = _MockLocalCritic(should_raise=True)

        result = await run_critic(state, mock_llm, local_critic=local_critic)

        verdict = result["critic_verdict"]
        assert verdict.passed is True
        assert verdict.score == pytest.approx(7.6, abs=1e-6)
        assert len(local_critic.calls) == 1
        mock_llm.with_structured_output.assert_called_once_with(CriticOutput)
        assert result["reasoning_trace"][0].metadata["source"] == "cloud_fallback"

    async def test_fallback_default_when_no_local_and_no_cloud(self) -> None:
        """Critic should return deterministic default when no model is available."""
        state = _make_state()

        result = await run_critic(state, llm=None, local_critic=None)

        verdict = result["critic_verdict"]
        assert verdict.passed is True
        assert verdict.score == pytest.approx(7.0, abs=1e-6)
        assert result["reasoning_trace"][0].metadata["source"] == "default_no_model"

    async def test_dict_state_missing_intent(self) -> None:
        """Should handle missing intent in dict state (strategy defaults to unknown)."""
        state = {
            "question": "What is LoRA?",
            "draft_answer": "LoRA is a fine-tuning method.",
            "retrieved_contexts": [],
        }
        mock_llm = _make_mock_llm(score=7.5)

        result = await run_critic(state, mock_llm)

        assert result["critic_verdict"].passed is True


# ---------------------------------------------------------------------------
# create_critic_node (factory)
# ---------------------------------------------------------------------------


class TestCreateCriticNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that processes state."""
        mock_llm = _make_mock_llm(score=8.0)

        node = create_critic_node(mock_llm)
        state = _make_state()

        result = await node(state)

        assert "critic_verdict" in result
        assert result["critic_verdict"].passed is True

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_llm = _make_mock_llm()

        node = create_critic_node(mock_llm)

        assert node.__name__ == "critic_node"

    async def test_factory_delegates_to_run_critic(self) -> None:
        """Factory node should produce same result as run_critic."""
        mock_llm = _make_mock_llm(score=5.5, feedback="Needs improvement.")

        node = create_critic_node(mock_llm)
        state = _make_state()

        result = await node(state)

        assert result["critic_verdict"].passed is False
        assert result["critic_verdict"].score == 5.5
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# critic_node (default sync placeholder)
# ---------------------------------------------------------------------------


class TestCriticNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_passing_verdict(self) -> None:
        """Placeholder should return a passing CriticVerdict."""
        state = _make_state()
        result = critic_node(state)

        assert "critic_verdict" in result
        verdict = result["critic_verdict"]
        assert isinstance(verdict, CriticVerdict)
        assert verdict.passed is True
        assert verdict.score == 7.0

    def test_placeholder_has_feedback(self) -> None:
        """Placeholder should include feedback text."""
        state = _make_state()
        result = critic_node(state)

        assert len(result["critic_verdict"].feedback) > 0

    def test_placeholder_has_quality_dimensions(self) -> None:
        """Placeholder should include completeness and faithfulness."""
        state = _make_state()
        result = critic_node(state)

        verdict = result["critic_verdict"]
        assert 0.0 <= verdict.completeness <= 1.0
        assert 0.0 <= verdict.faithfulness <= 1.0
