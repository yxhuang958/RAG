from __future__ import annotations

import httpx
from typing import Any

from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.endpoint = settings.llm_endpoint
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

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
                            "Answer only from supplied context, cite evidence IDs for factual claims, "
                            "and say you do not know when evidence is insufficient."
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
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, json=payload, headers=headers) # llm request
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text.strip()
                raise RuntimeError(f"LLM request failed: {exc}; response={detail}") from exc
            data = response.json()

        return self._extract_text(data)
