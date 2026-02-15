"""Conditional edges: route_by_intent, critic_gate, etc."""


def route_by_intent(state) -> str:
    """Route to the appropriate strategy based on intent type.

    Args:
        state: The current AgentState.

    Returns:
        The name of the target strategy node.
    """
    # Placeholder — implemented in task C3
    return "simple"


def critic_gate(state) -> str:
    """Decide whether to pass or retry based on critic verdict.

    Args:
        state: The current AgentState.

    Returns:
        ``"pass"`` or ``"retry"``.
    """
    # Placeholder — implemented in task D2
    return "pass"
