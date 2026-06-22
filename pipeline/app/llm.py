from __future__ import annotations

import httpx
import time
from typing import Any

from .config import get_settings
from .sanitize import redact_secrets


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)
_RETRYABLE_ERRORS = _RETRYABLE_HTTP_ERRORS + (RuntimeError,)


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.endpoint = settings.llm_endpoint
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        self.retries = max(1, settings.llm_retries)

    def _uses_chat_completions(self) -> bool:
        provider = (self.provider or "").lower()
        endpoint = self.endpoint or ""
        return provider in {"deepseek", "openai", "chat"} or endpoint.endswith("/chat/completions")

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        if self._uses_chat_completions():
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded RAG assistant. "
                            "Use supplied context as the evidence source, cite evidence IDs for factual claims, "
                            "and make a best-effort answer whenever the context contains relevant clues. "
                            "Say you do not know only when the context is empty or clearly unrelated."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            return payload

        return {"model": self.model, "prompt": prompt}

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                return message.get("content") or ""
            text = choices[0].get("text") if isinstance(choices[0], dict) else None
            if isinstance(text, str):
                return text

        return data.get("answer") or data.get("text") or ""

    def generate(self, prompt: str) -> str:
        if self.provider == "mock" or not self.endpoint:
            return "[MOCK ANSWER] " + prompt[:200]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(prompt)
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(1, self.retries + 1):
                try:
                    response = client.post(self.endpoint, json=payload, headers=headers) # llm request
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        detail = redact_secrets(response.text.strip())
                        raise RuntimeError(
                            f"retryable LLM HTTP {response.status_code}: {detail[:500]}"
                        )

                    response.raise_for_status()
                    data = response.json()
                    return self._extract_text(data)
                except httpx.HTTPStatusError as exc:
                    detail = redact_secrets(exc.response.text.strip())
                    error = redact_secrets(exc)
                    raise RuntimeError(f"LLM request failed: {error}; response={detail}") from exc
                except _RETRYABLE_ERRORS as exc:
                    last_error = exc
                    if attempt >= self.retries:
                        break
                    wait_seconds = min(8.0, 0.5 * (2 ** (attempt - 1)))
                    print(
                        f"WARNING - LLM request attempt {attempt}/{self.retries} failed; "
                        f"retrying in {wait_seconds:.1f}s: {redact_secrets(exc)}",
                        flush=True,
                    )
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"LLM request failed after {self.retries} attempt(s): {redact_secrets(last_error)}"
        ) from last_error
