"""Inference helpers for local Router/Critic models.

This module focuses on parsing and prompt-building utilities so model-loading
logic can stay isolated in ``src.models.loader``.
"""

from __future__ import annotations

import json
import re
from typing import Literal

IntentType = Literal[
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
]

_VALID_INTENTS: set[str] = {
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
}

_INTENT_ALIASES: dict[str, IntentType] = {
    "fact": "factual",
    "qa": "factual",
    "comparison": "comparative",
    "compare": "comparative",
    "multi-hop": "multi_hop",
    "multihop": "multi_hop",
    "explore": "exploratory",
    "followup": "follow_up",
    "follow-up": "follow_up",
}


def build_router_classification_prompt(question: str) -> str:
    """Build the Router prompt expected by local intent models.

    Args:
        question: User input question to classify.

    Returns:
        A deterministic prompt asking for strict JSON output.
    """
    return (
        "Classify the user question into one intent type.\n"
        "Allowed types: factual, comparative, multi_hop, exploratory, follow_up.\n"
        "Return ONLY valid JSON with keys: type, confidence.\n"
        'Example: {"type":"factual","confidence":0.92}\n'
        f"Question: {question.strip()}"
    )


def parse_router_classification(
    raw_text: str,
    *,
    default_intent: IntentType = "factual",
    default_confidence: float = 0.5,
) -> tuple[IntentType, float]:
    """Parse local Router output into ``(intent_type, confidence)``.

    Supports JSON-like output and loose ``type: ... confidence: ...`` text.

    Args:
        raw_text: Raw model response text.
        default_intent: Fallback intent when parsing fails.
        default_confidence: Fallback confidence when parsing fails.

    Returns:
        A tuple of normalized ``(intent_type, confidence)``.
    """
    text = raw_text.strip()
    if not text:
        return default_intent, _normalize_confidence(default_confidence)

    parsed = _parse_json_payload(text)
    if parsed:
        parsed_intent, parsed_conf = parsed
        return _normalize_intent(parsed_intent, default_intent), _normalize_confidence(parsed_conf)

    intent_match = re.search(r"(?:type|intent)\s*[:=]\s*([a-zA-Z_\-]+)", text, flags=re.IGNORECASE)
    conf_match = re.search(r"(?:confidence|score)\s*[:=]\s*([0-9]*\.?[0-9]+%?)", text, flags=re.IGNORECASE)

    intent = _normalize_intent(intent_match.group(1) if intent_match else "", default_intent)
    confidence = _normalize_confidence(conf_match.group(1) if conf_match else default_confidence)
    return intent, confidence


def _parse_json_payload(text: str) -> tuple[str, float | str] | None:
    """Extract JSON object from response text if present."""
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    raw_intent = payload.get("type", payload.get("intent"))
    raw_confidence = payload.get("confidence", payload.get("score", 0.5))
    if isinstance(raw_intent, str):
        return raw_intent, raw_confidence
    return None


def _normalize_intent(value: str, default: IntentType) -> IntentType:
    normalized = value.strip().lower().replace(" ", "_")
    normalized = _INTENT_ALIASES.get(normalized, normalized)
    if normalized in _VALID_INTENTS:
        return normalized  # type: ignore[return-value]
    return default


def _normalize_confidence(value: float | str) -> float:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            value = value[:-1]
            try:
                return max(0.0, min(float(value) / 100.0, 1.0))
            except ValueError:
                return 0.5
        try:
            value = float(value)
        except ValueError:
            return 0.5

    confidence = float(value)
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(confidence, 1.0))
"""Router / Critic inference wrappers for local models."""


def placeholder() -> None:
    """Placeholder — implemented in task F1."""
