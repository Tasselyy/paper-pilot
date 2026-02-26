"""Unit tests for src/models/vllm_client.py."""

from __future__ import annotations

import types

import pytest

from src.models.vllm_client import VLLMInferenceClient


class _FakeChoice:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.choices = [_FakeChoice(text)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._responses: list[str] = []

    def enqueue(self, text: str) -> None:
        self._responses.append(text)

    def create(self, **kwargs: object) -> _FakeCompletion:
        self.calls.append(kwargs)
        return _FakeCompletion(self._responses.pop(0))


class _FakeOpenAIClient:
    def __init__(self, *_: object, **__: object) -> None:
        self.completions = _FakeCompletions()


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIClient:
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIClient)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    return _FakeOpenAIClient()


def test_classify_question_parses_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_openai(monkeypatch)
    client = VLLMInferenceClient(router_model="router-lora", critic_model="critic-lora")
    client._client = fake_client  # type: ignore[assignment]
    fake_client.completions.enqueue('{"type":"comparative","confidence":0.88}')
    intent, confidence = client.classify_question("Compare LoRA and QLoRA")
    assert intent == "comparative"
    assert confidence == pytest.approx(0.88, abs=1e-6)


def test_evaluate_answer_parses_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_openai(monkeypatch)
    client = VLLMInferenceClient(router_model="router-lora", critic_model="critic-lora")
    client._client = fake_client  # type: ignore[assignment]
    fake_client.completions.enqueue(
        '{"score":8.4,"completeness":0.83,"faithfulness":0.9,"feedback":"Well grounded."}'
    )
    result = client.evaluate_answer(
        question="What is LoRA?",
        draft_answer="LoRA is a PEFT method.",
        retrieved_contexts=[],
        strategy="simple",
    )
    assert result["passed"] is True
    assert result["score"] == pytest.approx(8.4, abs=1e-6)
    assert result["completeness"] == pytest.approx(0.83, abs=1e-6)


def test_missing_router_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    client = VLLMInferenceClient(router_model=None, critic_model="critic-lora")
    with pytest.raises(ValueError):
        client.classify_question("What is LoRA?")
