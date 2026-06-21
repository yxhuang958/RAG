from __future__ import annotations

from typing import List


def build_prompt(messages: list[dict], contexts: List[dict]) -> str:
    context_blocks = []
    for idx, ctx in enumerate(contexts, start=1):
        doc_id = ctx.get("doc_id") or "unknown_doc"
        chunk_index = ctx.get("chunk_index", idx - 1)
        context_blocks.append(
            f"[{idx}] doc_id={doc_id} chunk_index={chunk_index}\n{ctx['text']}"
        )

    context_str = "\n\n".join(context_blocks)
    history_blocks = []
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_blocks.append(f"{role}: {msg['content']}")

    history_str = "\n".join(history_blocks) if history_blocks else "(No previous conversation.)"
    question = messages[-1]["content"]

    return (
        "Task: Answer the question using the provided RAG context.\n\n"
        "Guidelines:\n"
        "1. Prefer facts from the Context. Multi-hop conclusions are allowed when they follow from the retrieved passages.\n"
        "2. Avoid unsupported guesses. If the context is clearly insufficient, answer: "
        "\"I don't know based on the provided context.\"\n"
        "3. Keep the Answer line concise, especially for factoid questions.\n"
        "4. Add brief Evidence bullets when helpful, and include context IDs like [1] or [2] when citing specific passages.\n"
        "5. Use conversation history only to resolve references; the Context is the source of evidence.\n\n"
        "Preferred output format:\n"
        "Answer: <short answer only>\n"
        "Evidence: <optional brief supporting notes with context IDs>\n\n"
        f"Context:\n{context_str}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Question: {question}"
    )
