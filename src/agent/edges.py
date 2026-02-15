"""Conditional edges: route_by_intent, critic_gate, etc.

These functions are used by ``add_conditional_edges`` in the main graph
to decide which node to visit next based on the current state.
"""


def route_by_intent(state) -> str:
    """Route to the appropriate strategy based on intent type.

    Uses ``state.intent.to_strategy()`` to map the classified intent to
    a strategy node name.  ``follow_up`` is mapped to ``"simple"``.

    Args:
        state: The current AgentState (dict-like).

    Returns:
        One of ``"simple"``, ``"multi_hop"``, ``"comparative"``,
        ``"exploratory"``.
    """
    intent = state.get("intent") if isinstance(state, dict) else getattr(state, "intent", None)
    if intent is None:
        return "simple"
    return intent.to_strategy()


def critic_gate(state) -> str:
    """Decide whether to pass or retry based on critic verdict.

    Args:
        state: The current AgentState (dict-like).

    Returns:
        ``"pass"`` when the critic approves or max retries reached,
        ``"retry"`` otherwise.
    """
    verdict = state.get("critic_verdict") if isinstance(state, dict) else getattr(state, "critic_verdict", None)
    if verdict is None or verdict.passed:
        return "pass"

    retry_count = state.get("retry_count", 0) if isinstance(state, dict) else getattr(state, "retry_count", 0)
    max_retries = state.get("max_retries", 2) if isinstance(state, dict) else getattr(state, "max_retries", 2)
    if retry_count >= max_retries:
        return "pass"

    return "retry"
