"""vLLM OpenAI-compatible inference client for Router and Critic protocols."""

from __future__ import annotations

from typing import Any

from src.agent.state import IntentType
from src.models.inference import (
    build_critic_evaluation_prompt,
    build_router_classification_prompt,
    format_contexts_for_local_critic,
    parse_critic_evaluation,
    parse_router_classification,
)


class VLLMInferenceClient:
    """Unified local inference client for vLLM Multi-LoRA serving."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-placeholder",
        router_model: str | None = None,
        critic_model: str | None = None,
        max_new_tokens_router: int = 64,
        max_new_tokens_critic: int = 160,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("Missing vLLM client dependency. Install with: pip install -e .[vllm]") from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.router_model = router_model
        self.critic_model = critic_model
        self.max_new_tokens_router = max_new_tokens_router
        self.max_new_tokens_critic = max_new_tokens_critic

    def classify_question(self, question: str) -> tuple[IntentType, float]:
        """Classify question intent via router LoRA model served by vLLM."""
        if not self.router_model:
            raise ValueError("router_model is required for classify_question()")
        if not question.strip():
            return "factual", 0.0

        prompt = build_router_classification_prompt(question)
        completion = self._client.completions.create(
            model=self.router_model,
            prompt=prompt,
            max_tokens=self.max_new_tokens_router,
            temperature=0.0,
        )
        text = self._extract_text(completion)
        return parse_router_classification(text)

    def evaluate_answer(
        self,
        *,
        question: str,
        draft_answer: str,
        retrieved_contexts: list[Any] | None = None,
        strategy: str = "unknown",
        pass_threshold: float = 7.0,
    ) -> dict[str, float | bool | str]:
        """Evaluate answer quality via critic LoRA model served by vLLM."""
        if not self.critic_model:
            raise ValueError("critic_model is required for evaluate_answer()")
        if not draft_answer.strip():
            return {
                "passed": False,
                "score": 0.0,
                "completeness": 0.0,
                "faithfulness": 0.0,
                "feedback": "Draft answer is empty. The strategy must provide a substantive answer.",
            }

        contexts = format_contexts_for_local_critic(retrieved_contexts or [])
        prompt = build_critic_evaluation_prompt(
            question=question,
            draft_answer=draft_answer,
            contexts=contexts,
            strategy=strategy,
        )
        completion = self._client.completions.create(
            model=self.critic_model,
            prompt=prompt,
            max_tokens=self.max_new_tokens_critic,
            temperature=0.0,
        )
        text = self._extract_text(completion)
        score, completeness, faithfulness, feedback = parse_critic_evaluation(text)
        return {
            "passed": score >= pass_threshold,
            "score": score,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "feedback": feedback,
        }

    @staticmethod
    def _extract_text(completion: Any) -> str:
        """Best-effort extraction from OpenAI completion payload."""
        choices = getattr(completion, "choices", None)
        if choices and len(choices) > 0:
            text = getattr(choices[0], "text", "")
            if isinstance(text, str):
                return text
        return ""
