"""Unit tests for src/agent/edges.py — route_by_intent, critic_gate.

Verifies that all 5 ``IntentType`` values are routed to the correct
strategy branch, and that ``critic_gate`` correctly decides pass/retry.

Design reference: DEV_SPEC C3.
"""

from __future__ import annotations

import pytest

from src.agent.edges import (
    _DEFAULT_STRATEGY,
    _VALID_STRATEGIES,
    _get,
    critic_gate,
    route_by_intent,
)
from src.agent.state import AgentState, CriticVerdict, Intent, IntentType, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(intent_type: IntentType, confidence: float = 0.9) -> Intent:
    """Create an ``Intent`` instance with sensible defaults."""
    return Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=f"test query for {intent_type}",
    )


def _make_verdict(*, passed: bool, score: float = 5.0) -> CriticVerdict:
    """Create a ``CriticVerdict`` instance."""
    return CriticVerdict(
        passed=passed,
        score=score,
        completeness=0.8,
        faithfulness=0.8,
        feedback="" if passed else "Needs improvement",
    )


# ===========================================================================
# _get helper
# ===========================================================================


class TestGetHelper:
    """Tests for the ``_get`` state accessor."""

    def test_get_from_dict(self) -> None:
        state = {"intent": "value", "retry_count": 3}
        assert _get(state, "intent") == "value"
        assert _get(state, "retry_count") == 3

    def test_get_from_dict_missing_returns_default(self) -> None:
        state: dict = {}
        assert _get(state, "intent") is None
        assert _get(state, "retry_count", 0) == 0

    def test_get_from_model(self) -> None:
        state = AgentState(question="hello", retry_count=2)
        assert _get(state, "question") == "hello"
        assert _get(state, "retry_count") == 2

    def test_get_from_model_default(self) -> None:
        state = AgentState()
        assert _get(state, "intent") is None
        assert _get(state, "retry_count") == 0


# ===========================================================================
# route_by_intent
# ===========================================================================


class TestRouteByIntent:
    """Tests for ``route_by_intent`` conditional edge."""

    # -- 5 intent types → correct strategy ----------------------------------

    def test_factual_routes_to_simple(self) -> None:
        """``factual`` intent type should route to ``simple``."""
        state = {"intent": _make_intent("factual")}
        assert route_by_intent(state) == "simple"

    def test_follow_up_routes_to_simple(self) -> None:
        """``follow_up`` intent type should route to ``simple``."""
        state = {"intent": _make_intent("follow_up")}
        assert route_by_intent(state) == "simple"

    def test_multi_hop_routes_to_multi_hop(self) -> None:
        """``multi_hop`` intent type should route to ``multi_hop``."""
        state = {"intent": _make_intent("multi_hop")}
        assert route_by_intent(state) == "multi_hop"

    def test_comparative_routes_to_comparative(self) -> None:
        """``comparative`` intent type should route to ``comparative``."""
        intent = Intent(
            type="comparative",
            confidence=0.92,
            entities=["BERT", "GPT"],
            dimensions=["architecture", "performance"],
            reformulated_query="Compare BERT and GPT",
        )
        state = {"intent": intent}
        assert route_by_intent(state) == "comparative"

    def test_exploratory_routes_to_exploratory(self) -> None:
        """``exploratory`` intent type should route to ``exploratory``."""
        state = {"intent": _make_intent("exploratory")}
        assert route_by_intent(state) == "exploratory"

    # -- Comprehensive mapping -----------------------------------------------

    @pytest.mark.parametrize(
        ("intent_type", "expected_strategy"),
        [
            ("factual", "simple"),
            ("follow_up", "simple"),
            ("multi_hop", "multi_hop"),
            ("comparative", "comparative"),
            ("exploratory", "exploratory"),
        ],
    )
    def test_all_five_intent_types_to_correct_strategy(
        self,
        intent_type: IntentType,
        expected_strategy: StrategyName,
    ) -> None:
        """All 5 ``IntentType`` values must route to the correct strategy."""
        state = {"intent": _make_intent(intent_type)}
        result = route_by_intent(state)
        assert result == expected_strategy, (
            f"IntentType '{intent_type}' should route to '{expected_strategy}', "
            f"got '{result}'"
        )

    # -- Return value is a valid strategy ------------------------------------

    @pytest.mark.parametrize("intent_type", ["factual", "follow_up", "multi_hop", "comparative", "exploratory"])
    def test_return_value_in_valid_strategies(self, intent_type: IntentType) -> None:
        """Return value must always be a member of ``_VALID_STRATEGIES``."""
        state = {"intent": _make_intent(intent_type)}
        assert route_by_intent(state) in _VALID_STRATEGIES

    # -- Fallback behaviour --------------------------------------------------

    def test_none_intent_defaults_to_simple(self) -> None:
        """When ``intent`` is ``None``, fall back to ``simple``."""
        state: dict = {"intent": None}
        assert route_by_intent(state) == _DEFAULT_STRATEGY

    def test_missing_intent_key_defaults_to_simple(self) -> None:
        """When ``intent`` key is missing entirely, fall back to ``simple``."""
        state: dict = {}
        assert route_by_intent(state) == _DEFAULT_STRATEGY

    # -- Works with AgentState model instances --------------------------------

    def test_route_with_agent_state_model_factual(self) -> None:
        """``route_by_intent`` works with an ``AgentState`` Pydantic model."""
        state = AgentState(
            question="What is LoRA?",
            intent=_make_intent("factual"),
        )
        assert route_by_intent(state) == "simple"

    def test_route_with_agent_state_model_comparative(self) -> None:
        state = AgentState(
            question="Compare BERT and GPT",
            intent=_make_intent("comparative"),
        )
        assert route_by_intent(state) == "comparative"

    def test_route_with_agent_state_model_multi_hop(self) -> None:
        state = AgentState(
            question="How does LoRA relate to quantisation?",
            intent=_make_intent("multi_hop"),
        )
        assert route_by_intent(state) == "multi_hop"

    def test_route_with_agent_state_model_exploratory(self) -> None:
        state = AgentState(
            question="Recent advances in LLM training?",
            intent=_make_intent("exploratory"),
        )
        assert route_by_intent(state) == "exploratory"

    def test_route_with_agent_state_model_follow_up(self) -> None:
        state = AgentState(
            question="Tell me more about that",
            intent=_make_intent("follow_up"),
        )
        assert route_by_intent(state) == "simple"

    def test_route_with_agent_state_model_no_intent(self) -> None:
        """``AgentState`` with default ``intent=None`` → ``simple``."""
        state = AgentState(question="fallback question")
        assert route_by_intent(state) == _DEFAULT_STRATEGY

    # -- Confidence values don't affect routing ------------------------------

    def test_low_confidence_does_not_change_route(self) -> None:
        """Strategy routing is purely by type, confidence doesn't affect it."""
        state = {"intent": _make_intent("multi_hop", confidence=0.1)}
        assert route_by_intent(state) == "multi_hop"

    def test_high_confidence_does_not_change_route(self) -> None:
        state = {"intent": _make_intent("factual", confidence=1.0)}
        assert route_by_intent(state) == "simple"


# ===========================================================================
# critic_gate
# ===========================================================================


class TestCriticGate:
    """Tests for ``critic_gate`` conditional edge."""

    # -- Pass conditions ----------------------------------------------------

    def test_no_verdict_returns_pass(self) -> None:
        """No ``critic_verdict`` → ``"pass"``."""
        state: dict = {}
        assert critic_gate(state) == "pass"

    def test_none_verdict_returns_pass(self) -> None:
        state: dict = {"critic_verdict": None}
        assert critic_gate(state) == "pass"

    def test_passed_verdict_returns_pass(self) -> None:
        """Critic approved → ``"pass"``."""
        state = {"critic_verdict": _make_verdict(passed=True, score=8.5)}
        assert critic_gate(state) == "pass"

    # -- Retry conditions ---------------------------------------------------

    def test_failed_verdict_below_max_retries_returns_retry(self) -> None:
        """Critic failed, retries remaining → ``"retry"``."""
        state = {
            "critic_verdict": _make_verdict(passed=False, score=3.0),
            "retry_count": 0,
            "max_retries": 2,
        }
        assert critic_gate(state) == "retry"

    def test_failed_verdict_one_retry_done_returns_retry(self) -> None:
        state = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 1,
            "max_retries": 2,
        }
        assert critic_gate(state) == "retry"

    # -- Max retries reached → force pass -----------------------------------

    def test_failed_verdict_at_max_retries_returns_pass(self) -> None:
        """Critic failed but max retries reached → ``"pass"`` (forced)."""
        state = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 2,
            "max_retries": 2,
        }
        assert critic_gate(state) == "pass"

    def test_failed_verdict_beyond_max_retries_returns_pass(self) -> None:
        state = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 5,
            "max_retries": 2,
        }
        assert critic_gate(state) == "pass"

    # -- Default max_retries ------------------------------------------------

    def test_default_max_retries_is_two(self) -> None:
        """When ``max_retries`` is not set, default should be 2."""
        state = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 0,
        }
        assert critic_gate(state) == "retry"

        state_at_limit = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 2,
        }
        assert critic_gate(state_at_limit) == "pass"

    # -- Works with AgentState model ----------------------------------------

    def test_critic_gate_with_agent_state_model_pass(self) -> None:
        state = AgentState(
            question="q",
            critic_verdict=_make_verdict(passed=True),
        )
        assert critic_gate(state) == "pass"

    def test_critic_gate_with_agent_state_model_retry(self) -> None:
        state = AgentState(
            question="q",
            critic_verdict=_make_verdict(passed=False),
            retry_count=0,
            max_retries=2,
        )
        assert critic_gate(state) == "retry"

    def test_critic_gate_with_agent_state_model_max_reached(self) -> None:
        state = AgentState(
            question="q",
            critic_verdict=_make_verdict(passed=False),
            retry_count=2,
            max_retries=2,
        )
        assert critic_gate(state) == "pass"

    def test_critic_gate_agent_state_no_verdict(self) -> None:
        state = AgentState(question="q")
        assert critic_gate(state) == "pass"

    # -- Edge cases ---------------------------------------------------------

    def test_zero_max_retries_always_passes(self) -> None:
        """If ``max_retries`` is 0, critic_gate should always pass."""
        state = {
            "critic_verdict": _make_verdict(passed=False),
            "retry_count": 0,
            "max_retries": 0,
        }
        assert critic_gate(state) == "pass"
