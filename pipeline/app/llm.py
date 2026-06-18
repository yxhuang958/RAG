from __future__ import annotations

import httpx

from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.endpoint = settings.llm_endpoint
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    def generate(self, prompt: str) -> str:
        if self.provider == "mock" or not self.endpoint:
            return "[MOCK ANSWER] " + prompt[:200]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "prompt": prompt}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, json=payload, headers=headers) # llm request
            response.raise_for_status()
            data = response.json()

        return data.get("answer") or data.get("text") or ""
