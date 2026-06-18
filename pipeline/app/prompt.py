from __future__ import annotations

from typing import List


def build_prompt(messages: list[dict], contexts: List[dict]) -> str:
    context_blocks = []
    for idx, ctx in enumerate(contexts, start=1):
        context_blocks.append(f"[{idx}] {ctx['text']}")

    context_str = "\n\n".join(context_blocks)
    history_blocks = []
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_blocks.append(f"{role}: {msg['content']}")

    history_str = "\n".join(history_blocks) if history_blocks else "(No previous conversation.)"
    question = messages[-1]["content"]

    return (
        "You are a helpful assistant. Answer the question using the provided context.\n"
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Question: {question}\nAnswer:"
    )
