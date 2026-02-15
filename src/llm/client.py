"""Cloud LLM client abstraction (OpenAI-compatible).

Provides ``create_llm`` — a config-driven factory that returns a
``BaseChatModel`` based on the ``LLMConfig`` from application settings.

Currently supports the **openai** provider (via ``ChatOpenAI``).  Extend
the factory when additional providers are needed (e.g. Azure, local).

Design reference: DEV_SPEC §3.1, §5.5.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.config import LLMConfig

logger = logging.getLogger(__name__)


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create a ``BaseChatModel`` from application LLM configuration.

    Reads the provider, model name, temperature, and API key from the
    validated ``LLMConfig`` and returns the appropriate LangChain chat
    model instance.

    Args:
        config: Validated ``LLMConfig`` from ``load_settings().llm``.

    Returns:
        A configured ``BaseChatModel`` instance ready for ``ainvoke``.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider == "openai":
        kwargs: dict = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key

        llm = ChatOpenAI(**kwargs)
        logger.info(
            "Created ChatOpenAI (model=%s, temperature=%.2f)",
            config.model,
            config.temperature,
        )
        return llm

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. Supported: openai"
    )
