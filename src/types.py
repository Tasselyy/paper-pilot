"""Shared type aliases with no heavy dependencies.

Used by both agent and models so that code that only needs inference/types
(e.g. training data generation) does not pull in the full agent graph,
LangChain, or transformers.
"""

from __future__ import annotations

from typing import Literal

IntentType = Literal[
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
]
"""Closed-set intent types produced by the Router (LoRA classifier)."""
