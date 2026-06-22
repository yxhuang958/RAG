from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from app.sanitize import redact_secrets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def extract_openai_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts)


def main() -> None:
    load_env_file(ENV_PATH)

    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/responses")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    if not api_key:
        raise SystemExit("Missing LLM_API_KEY in .env")

    prompt = "Reply with one short sentence: the LLM API is working."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if endpoint.endswith("/chat/completions"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif endpoint.endswith("/responses"):
        payload = {"model": model, "input": prompt}
    else:
        payload = {"model": model, "prompt": prompt}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, headers=headers, json=payload)

    if response.is_error:
        print(f"LLM API test failed: HTTP {response.status_code}")
        print(redact_secrets(response.text))
        raise SystemExit(1)

    data = response.json()
    text = (
        extract_openai_text(data)
        or data.get("answer")
        or data.get("text")
        or json.dumps(data, ensure_ascii=False)[:1000]
    )

    print("LLM API test: ok")
    print(text)


if __name__ == "__main__":
    main()
