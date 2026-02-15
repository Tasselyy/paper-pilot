"""Memory nodes: load_memory and save_memory.

Placeholder implementations — no-op pass-through.
Real memory logic is wired in task D4.
"""

from __future__ import annotations


def load_memory_node(state) -> dict:
    """Load relevant facts from long-term memory into state.

    Args:
        state: The current AgentState.

    Returns:
        Empty dict (placeholder).
    """
    return {}


def save_memory_node(state) -> dict:
    """Persist notable facts from the current turn to long-term memory.

    Args:
        state: The current AgentState.

    Returns:
        Empty dict (placeholder).
    """
    return {}
