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
        "Task: Answer the question using only the provided RAG context.\n\n"
        "Grounding rules:\n"
        "1. Use only facts that are directly supported by the Context section.\n"
        "2. Do not use prior knowledge, guesses, or unstated assumptions.\n"
        "3. Cite every factual claim with context IDs like [1] or [2].\n"
        "4. If the context does not contain enough evidence, answer exactly: "
        "\"I don't know based on the provided context.\"\n"
        "5. If context passages conflict, state the conflict and cite the conflicting passages instead of guessing.\n"
        "6. Conversation history is only for resolving references; it is not evidence unless repeated in Context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Question: {question}\nAnswer:"
    )
