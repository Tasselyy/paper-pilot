"""Critic node: evaluate answer quality via DPO model or Cloud LLM."""


async def critic_node(state):
    """Evaluate the draft answer and produce a CriticVerdict.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``critic_verdict``.
    """
    # Placeholder — implemented in task D1
    return state
