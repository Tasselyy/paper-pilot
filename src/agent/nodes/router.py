"""Router node: intent classification via LoRA model or Cloud LLM fallback."""


async def router_node(state):
    """Classify the user question into an intent type.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``intent.type`` and ``intent.confidence``.
    """
    # Placeholder — implemented in task C1
    return state
