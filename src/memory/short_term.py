"""Short-term memory: LangGraph Checkpointer wrapper.

Provides a thin factory around LangGraph's built-in checkpointer backends.
Short-term memory (conversation messages, in-flight ``AgentState``) is
automatically managed by LangGraph's ``MemorySaver`` (in-memory) or
persistent savers (SQLite, Postgres).

This module exposes ``create_checkpointer`` so the rest of the codebase
has a single place to configure the checkpointer backend.

Design reference: PAPER_PILOT_DESIGN.md §8.1.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

CheckpointerBackend = Literal["memory"]
"""Supported checkpointer backends.

Currently only ``"memory"`` is implemented.  ``"sqlite"`` and
``"postgres"`` can be added later for production persistence.
"""


def create_checkpointer(
    backend: CheckpointerBackend = "memory",
) -> BaseCheckpointSaver:
    """Create and return a LangGraph checkpointer instance.

    Args:
        backend: The checkpointer backend to use.  Currently only
            ``"memory"`` is supported.

    Returns:
        A ``BaseCheckpointSaver`` instance.

    Raises:
        ValueError: If an unsupported backend is specified.
    """
    if backend == "memory":
        logger.info("Using in-memory checkpointer (MemorySaver)")
        return MemorySaver()

    raise ValueError(
        f"Unsupported checkpointer backend: {backend!r}. "
        f"Supported: 'memory'."
    )
