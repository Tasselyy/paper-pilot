"""Memory nodes: load_memory and save_memory."""


async def load_memory_node(state):
    """Load relevant facts from long-term memory into state.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``accumulated_facts``.
    """
    # Placeholder — implemented in task D4
    return state


async def save_memory_node(state):
    """Persist notable facts from the current turn to long-term memory.

    Args:
        state: The current AgentState.

    Returns:
        Unchanged state.
    """
    # Placeholder — implemented in task D4
    return state
