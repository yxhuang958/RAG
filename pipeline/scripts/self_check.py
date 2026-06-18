from app.prompt import build_prompt
from app.text_utils import chunk_text


def main() -> None:
    text = "A" * 1200
    chunks = chunk_text(text, 500, 100)
    assert len(chunks) >= 2, "chunking should create multiple chunks"

    prompt = build_prompt("What is A?", [{"text": "AAA"}])
    assert "Context" in prompt and "Question" in prompt

    print("self_check: ok")


if __name__ == "__main__":
    main()
