"""Inference helpers for local Router/Critic models.

This module focuses on parsing and prompt-building utilities so model-loading
logic can stay isolated in ``src.models.loader``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

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


def build_critic_evaluation_prompt(
    question: str,
    draft_answer: str,
    contexts: str,
    *,
    strategy: str = "unknown",
) -> str:
    """Build the Critic prompt expected by local quality-evaluation models.

    Args:
        question: User's original question.
        draft_answer: Current answer draft to evaluate.
        contexts: Retrieved contexts formatted as plain text.
        strategy: Strategy label used to produce the draft answer.

    Returns:
        A deterministic prompt asking for strict JSON output.
    """
    return (
        "Evaluate the answer quality for the given question and retrieved context.\n"
        "Score rubric:\n"
        "- score: overall quality in [0, 10]\n"
        "- completeness: coverage in [0, 1]\n"
        "- faithfulness: grounding in context in [0, 1]\n"
        "Return ONLY valid JSON with keys: score, completeness, faithfulness, feedback.\n"
        '{"score":7.8,"completeness":0.82,"faithfulness":0.86,'
        '"feedback":"Add a concrete example."}\n'
        f"Strategy: {strategy.strip() or 'unknown'}\n"
        f"Question: {question.strip()}\n"
        f"Draft Answer:\n{draft_answer.strip()}\n"
        f"Contexts:\n{contexts.strip()}"
    )


def format_contexts_for_local_critic(retrieved_contexts: list[Any]) -> str:
    """Convert retrieved contexts into deterministic plain text.

    Args:
        retrieved_contexts: Context objects or strings from retrieval.

    Returns:
        A numbered text block suitable for local prompt inputs.
    """
    if not retrieved_contexts:
        return "(no retrieved contexts)"

    normalized: list[str] = []
    for index, item in enumerate(retrieved_contexts, start=1):
        if isinstance(item, str):
            content = item.strip()
        elif isinstance(item, dict):
            content = str(
                item.get("content")
                or item.get("text")
                or item.get("snippet")
                or item
            ).strip()
        else:
            content = str(item).strip()

        if not content:
            continue
        normalized.append(f"[{index}] {content}")

    if not normalized:
        return "(no retrieved contexts)"
    return "\n".join(normalized)


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


def parse_critic_evaluation(
    raw_text: str,
    *,
    default_score: float = 5.0,
    default_completeness: float = 0.5,
    default_faithfulness: float = 0.5,
    default_feedback: str = "Local critic output could not be parsed reliably.",
) -> tuple[float, float, float, str]:
    """Parse local Critic output into normalized quality metrics.

    Supports strict JSON output and a loose text form such as:
    ``score=7.5 completeness=82% faithfulness=0.9 feedback=...``.

    Args:
        raw_text: Raw model response text.
        default_score: Fallback overall score in [0, 10].
        default_completeness: Fallback completeness in [0, 1].
        default_faithfulness: Fallback faithfulness in [0, 1].
        default_feedback: Fallback feedback when parsing fails.

    Returns:
        A tuple of ``(score, completeness, faithfulness, feedback)``.
    """
    text = raw_text.strip()
    if not text:
        return (
            _normalize_score(default_score),
            _normalize_ratio(default_completeness),
            _normalize_ratio(default_faithfulness),
            default_feedback.strip(),
        )

    json_payload = _parse_json_object(text)
    if json_payload is not None:
        score = _normalize_score(json_payload.get("score", default_score))
        completeness = _normalize_ratio(
            json_payload.get("completeness", default_completeness)
        )
        faithfulness = _normalize_ratio(
            json_payload.get("faithfulness", default_faithfulness)
        )
        feedback = _normalize_feedback(json_payload.get("feedback"), default_feedback)
        return score, completeness, faithfulness, feedback

    score_match = re.search(
        r"(?:score|overall)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        text,
        flags=re.IGNORECASE,
    )
    completeness_match = re.search(
        r"(?:completeness|coverage)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        text,
        flags=re.IGNORECASE,
    )
    faithfulness_match = re.search(
        r"(?:faithfulness|grounding)\s*[:=]\s*([0-9]*\.?[0-9]+%?)",
        text,
        flags=re.IGNORECASE,
    )
    feedback_match = re.search(
        r"(?:feedback|comment)\s*[:=]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    score = _normalize_score(score_match.group(1) if score_match else default_score)
    completeness = _normalize_ratio(
        completeness_match.group(1) if completeness_match else default_completeness
    )
    faithfulness = _normalize_ratio(
        faithfulness_match.group(1) if faithfulness_match else default_faithfulness
    )
    feedback = _normalize_feedback(
        feedback_match.group(1) if feedback_match else None,
        default_feedback,
    )
    return score, completeness, faithfulness, feedback


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from text if available and valid."""
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
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

def _normalize_score(value: float | str | None) -> float:
    """Normalize score values to [0, 10]."""
    number = _parse_float(value)
    if number is None:
        return 5.0
    if 0.0 <= number <= 1.0:
        number *= 10.0
    if 10.0 < number <= 100.0 and float(number).is_integer():
        # Prefer interpreting plain percentages such as "78" as 7.8.
        number = number / 10.0
    return max(0.0, min(number, 10.0))


def _normalize_ratio(value: float | str | None) -> float:
    """Normalize ratio values to [0, 1]."""
    number = _parse_float(value)
    if number is None:
        return 0.5
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return max(0.0, min(number, 1.0))


def _parse_float(value: float | str | None) -> float | None:
    """Parse a float from numeric/percent-like input."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.endswith("%"):
            stripped = stripped[:-1]
            try:
                return float(stripped)
            except ValueError:
                return None
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_feedback(value: Any, default_feedback: str) -> str:
    """Return a clean feedback string with fallback."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_feedback.strip()
