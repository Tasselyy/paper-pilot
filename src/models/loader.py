"""Local model manager for quantized Router/Critic inference.

Loads optional local models defined in config (e.g. LoRA Router, DPO Critic).
When local-model dependencies are not installed, this module raises a clear
error so callers can fall back to Cloud LLM paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.state import IntentType
from src.models.inference import (
    build_router_classification_prompt,
    parse_router_classification,
)


class LocalModelUnavailableError(RuntimeError):
    """Raised when local model dependencies are unavailable."""


@dataclass(slots=True)
class LoadedGenerationPipeline:
    """Container for a text-generation pipeline and metadata."""

    pipeline: Any
    model_path: str
    quantized_4bit: bool


class LocalModelManager:
    """Manage loading and inference of optional local models.

    Args:
        router_model_path: Local path or HF id for Router model.
        critic_model_path: Local path or HF id for Critic model.
        max_new_tokens: Generation cap for local inference outputs.
        use_4bit_if_available: Use 4-bit quantization when CUDA is available.
    """

    def __init__(
        self,
        *,
        router_model_path: str | None = None,
        critic_model_path: str | None = None,
        max_new_tokens: int = 64,
        use_4bit_if_available: bool = True,
    ) -> None:
        self.router_model_path = router_model_path
        self.critic_model_path = critic_model_path
        self.max_new_tokens = max_new_tokens
        self.use_4bit_if_available = use_4bit_if_available
        self._router_pipeline: LoadedGenerationPipeline | None = None

    def load_router(self, *, force_reload: bool = False) -> LoadedGenerationPipeline:
        """Load Router model pipeline, preferring 4-bit on CUDA.

        Args:
            force_reload: Recreate pipeline even if cached.

        Returns:
            A loaded text-generation pipeline wrapper.

        Raises:
            ValueError: If ``router_model_path`` is missing.
            LocalModelUnavailableError: If transformers/torch are unavailable.
        """
        if not self.router_model_path:
            raise ValueError("router_model_path is required to load local router model")

        if self._router_pipeline is not None and not force_reload:
            return self._router_pipeline

        self._router_pipeline = self._load_text_generation_pipeline(self.router_model_path)
        return self._router_pipeline

    def classify_question(self, question: str) -> tuple[IntentType, float]:
        """Classify question intent using the local Router model.

        Args:
            question: User question to classify.

        Returns:
            A tuple of ``(intent_type, confidence)``.
        """
        if not question.strip():
            return "factual", 0.0

        router = self.load_router()
        prompt = build_router_classification_prompt(question)
        generated = router.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        response_text = self._extract_generated_text(generated)
        return parse_router_classification(response_text)

    @staticmethod
    def is_cuda_available() -> bool:
        """Return whether CUDA is available in current environment."""
        try:
            import torch
        except Exception:
            return False
        return bool(torch.cuda.is_available())

    def _load_text_generation_pipeline(self, model_path: str) -> LoadedGenerationPipeline:
        """Create a local text-generation pipeline with optional 4-bit config."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except Exception as exc:  # pragma: no cover - env dependent
            raise LocalModelUnavailableError(
                "Local-model dependencies are missing. Install extras: "
                "`pip install -e .[local-models]`"
            ) from exc

        use_4bit = self.use_4bit_if_available and self.is_cuda_available()
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            if use_4bit
            else None
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return LoadedGenerationPipeline(
            pipeline=pipe,
            model_path=model_path,
            quantized_4bit=use_4bit,
        )

    @staticmethod
    def _extract_generated_text(generated: Any) -> str:
        """Extract generated string from transformers pipeline response."""
        if isinstance(generated, list) and generated:
            first = generated[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                return text if isinstance(text, str) else str(text)
            return str(first)
        if isinstance(generated, str):
            return generated
        return str(generated)
"""LocalModelManager: 4-bit quantized model loading."""


def placeholder() -> None:
    """Placeholder — implemented in task F1."""
