"""Unit tests for local model loading and Router inference (F1)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from src.models.inference import parse_router_classification
from src.models.loader import (
    LocalModelManager,
)


def _cuda_available() -> bool:
    """Check CUDA availability without requiring torch in test env."""
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


class FakePipeline:
    """Small callable object emulating transformers text-generation pipeline."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"prompt": prompt, **kwargs})
        return [{"generated_text": self.response_text}]


def test_parse_router_classification_json_payload() -> None:
    """Parses JSON output from local model response."""
    intent, confidence = parse_router_classification(
        '{"type":"comparative","confidence":0.81}'
    )
    assert intent == "comparative"
    assert confidence == pytest.approx(0.81, abs=1e-6)


def test_parse_router_classification_percent_and_alias() -> None:
    """Supports alias intent names and percent confidence format."""
    intent, confidence = parse_router_classification(
        "intent=multi-hop confidence=87%"
    )
    assert intent == "multi_hop"
    assert confidence == pytest.approx(0.87, abs=1e-6)


def test_parse_router_classification_fallback_defaults() -> None:
    """Falls back to default values on malformed outputs."""
    intent, confidence = parse_router_classification(
        "unrelated response without schema",
        default_intent="factual",
        default_confidence=0.42,
    )
    assert intent == "factual"
    assert confidence == pytest.approx(0.42, abs=1e-6)


def test_load_router_requires_path() -> None:
    """Router loader requires configured model path."""
    manager = LocalModelManager(router_model_path=None)
    with pytest.raises(ValueError, match="router_model_path"):
        manager.load_router()


def test_classify_question_with_mocked_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classifies question from mocked local model pipeline output."""
    manager = LocalModelManager(router_model_path="local/router")
    fake_pipeline = FakePipeline('{"type":"exploratory","confidence":0.73}')

    def _fake_loader(_: str) -> Any:
        return type(
            "LoadedPipeline",
            (),
            {"pipeline": fake_pipeline, "model_path": "local/router", "quantized_4bit": True},
        )()

    monkeypatch.setattr(manager, "_load_text_generation_pipeline", _fake_loader)

    intent_type, confidence = manager.classify_question("Survey efficient fine-tuning methods")
    assert intent_type == "exploratory"
    assert confidence == pytest.approx(0.73, abs=1e-6)
    assert fake_pipeline.calls, "Pipeline should be invoked once"

    call = fake_pipeline.calls[0]
    assert "Survey efficient fine-tuning methods" in call["prompt"]
    assert call["return_full_text"] is False
    assert call["do_sample"] is False


def test_classify_question_empty_input_short_circuit() -> None:
    """Empty questions should avoid model invocation and return deterministic default."""
    manager = LocalModelManager(router_model_path="unused")
    intent_type, confidence = manager.classify_question("")
    assert intent_type == "factual"
    assert confidence == 0.0


@pytest.mark.skipif(
    not (_cuda_available() and bool(os.getenv("LOCAL_ROUTER_MODEL_PATH"))),
    reason="Requires CUDA and LOCAL_ROUTER_MODEL_PATH for real 4-bit smoke test.",
)
def test_router_4bit_smoke_real_model() -> None:
    """Optional smoke test for 4-bit load in CUDA environments."""
    model_path = os.environ["LOCAL_ROUTER_MODEL_PATH"]
    manager = LocalModelManager(router_model_path=model_path)
    loaded = manager.load_router()
    assert loaded.quantized_4bit is True

