"""Conditional edges: route_by_intent, critic_gate, etc.

These functions are used by ``add_conditional_edges`` in the main graph
to decide which node to visit next based on the current state.

Design reference: PAPER_PILOT_DESIGN.md §5.1, DEV_SPEC C3.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from src.agent.state import AgentState, StrategyName

logger = logging.getLogger(__name__)

# Type alias for state — LangGraph may pass either a dict or model instance.
StateLike = Union[AgentState, dict[str, Any]]

# Valid strategy branch names (must match node names in graph.py)
_VALID_STRATEGIES: frozenset[str] = frozenset(
    {"simple", "multi_hop", "comparative", "exploratory"}
)

_DEFAULT_STRATEGY: StrategyName = "simple"


def _get(state: StateLike, key: str, default: Any = None) -> Any:
    """Extract a value from *state* regardless of dict or model form.

    Args:
        state: The current ``AgentState`` (dict or Pydantic model).
        key: Attribute / key name.
        default: Fallback when key is missing.

    Returns:
        The value associated with *key*, or *default*.
    """
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def route_by_intent(state: StateLike) -> StrategyName:
    """Route to the appropriate strategy based on intent type.

    Uses ``state.intent.to_strategy()`` to map the classified intent to
    a strategy node name.  ``follow_up`` is mapped to ``"simple"``.

    Falls back to ``"simple"`` when:
    * ``intent`` is ``None`` (no classification yet).
    * ``intent.to_strategy()`` returns an unrecognised name.

    Args:
        state: The current ``AgentState`` (dict or Pydantic model).

    Returns:
        One of ``"simple"``, ``"multi_hop"``, ``"comparative"``,
        ``"exploratory"``.
    """
    intent = _get(state, "intent")

    if intent is None:
        logger.warning("route_by_intent: intent is None — defaulting to '%s'", _DEFAULT_STRATEGY)
        return _DEFAULT_STRATEGY

    strategy: str = intent.to_strategy()

    if strategy not in _VALID_STRATEGIES:
        logger.warning(
            "route_by_intent: unrecognised strategy '%s' from intent type '%s' — defaulting to '%s'",
            strategy,
            getattr(intent, "type", "?"),
            _DEFAULT_STRATEGY,
        )
        return _DEFAULT_STRATEGY

    return strategy  # type: ignore[return-value]


def critic_gate(state: StateLike) -> str:
    """Decide whether to pass or retry based on critic verdict.

    Returns ``"pass"`` when:
    * No verdict exists (critic not yet run).
    * Verdict indicates the answer passed.
    * ``retry_count`` has reached ``max_retries``.

    Otherwise returns ``"retry"`` so the graph loops through
    ``retry_refine → route``.

    Args:
        state: The current ``AgentState`` (dict or Pydantic model).

    Returns:
        ``"pass"`` or ``"retry"``.
    """
    verdict = _get(state, "critic_verdict")

    if verdict is None or verdict.passed:
        return "pass"

    retry_count: int = _get(state, "retry_count", 0)
    max_retries: int = _get(state, "max_retries", 2)

    if retry_count >= max_retries:
        logger.info(
            "critic_gate: retry_count (%d) >= max_retries (%d) — forcing pass",
            retry_count,
            max_retries,
        )
        return "pass"

    return "retry"
