"""Thin OpenAI-compatible client for training data generation scripts.

Uses OPENAI_API_KEY and optional OPENAI_BASE_URL (e.g. for vLLM).
Requires: pip install openai  (or use extra: pip install -e ".[vllm]")
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_MODEL = "gpt-4o-mini"
# 单次请求超时（秒），避免卡死；可通过环境变量 OPENAI_TIMEOUT 覆盖
DEFAULT_TIMEOUT = 120.0


def _get_timeout() -> float:
    try:
        return float(os.environ.get("OPENAI_TIMEOUT", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def get_client() -> Any:
    """Build OpenAI client from env. Raises if openai not installed or API key missing."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "LLM data generation requires the openai package. "
            "Install with: pip install openai  or  pip install -e \".[vllm]\""
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url and not api_key:
        api_key = "dummy"
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the environment for LLM-based data generation. "
            "For vLLM, set OPENAI_BASE_URL and use a dummy OPENAI_API_KEY if needed."
        )
    timeout = _get_timeout()
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    except TypeError:
        return OpenAI(api_key=api_key, base_url=base_url)


def chat_text(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    max_retries: int = 2,
) -> tuple[str | None, str | None]:
    """Send one chat round and return the assistant reply as plain text.

    Returns:
        (content, None) on success, or (None, error_message) on failure.
    """
    if client is None:
        client = get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            return (text or None, None)
        except Exception as e:
            last_err = str(e)
    return (None, last_err)


def chat_json(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
    max_retries: int = 2,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send one chat round and parse the assistant reply as JSON.

    Returns:
        (parsed_dict, None) on success, or (None, error_message) on failure.
    """
    if client is None:
        client = get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = "Empty model response"
                continue
            obj = _extract_json(text)
            if obj is None:
                last_err = f"No valid JSON in response: {text[:200]}..."
                continue
            return (obj, None)
        except Exception as e:
            last_err = str(e)
    return (None, last_err)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to get a single JSON object from model output (handles markdown code blocks)."""
    # Strip markdown code block if present
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stripped)
    if match:
        stripped = match.group(1)
    # Find last {...} that spans a reasonable object
    depth = 0
    start = -1
    for i, c in enumerate(stripped):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(stripped[start : i + 1])
                except json.JSONDecodeError:
                    pass
    # Fallback: parse whole string
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
