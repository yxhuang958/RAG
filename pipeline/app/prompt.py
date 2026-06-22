from __future__ import annotations

from typing import List

from .query_preprocessor import ProcessedQuery


def build_prompt(
    messages: list[dict],
    contexts: List[dict],
    processed_query: ProcessedQuery | None = None,
) -> str:
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
        if msg["role"] == "system":
            role = "System"
        elif msg["role"] == "user":
            role = "User"
        else:
            role = "Assistant"
        history_blocks.append(f"{role}: {msg['content']}")

    history_str = "\n".join(history_blocks) if history_blocks else "(No previous conversation.)"
    question = processed_query.question if processed_query else messages[-1]["content"]
    temporary_context = (
        processed_query.temporary_context.strip()
        if processed_query and processed_query.temporary_context.strip()
        else ""
    )
    temporary_context_str = temporary_context or "(No temporary context provided.)"

    return (
        "Task: Answer the question using the provided RAG context.\n\n"
        "Guidelines:\n"
        "1. Prefer facts from the Context. Multi-hop conclusions are allowed when they follow from the retrieved passages.\n"
        "2. If the Context contains any relevant evidence, give the best supported answer. Do not answer with "
        "\"I don't know\" just because one bridge step is implicit or the wording is incomplete.\n"
        "3. Only answer \"I don't know based on the provided context\" when the Context is empty or clearly unrelated "
        "to the question.\n"
        "4. For factoid questions, make the Answer line the most specific entity, title, location, date, number, or "
        "yes/no conclusion supported by the Context.\n"
        "5. If evidence is partial, still answer from the strongest relevant clue and explain the uncertainty in Evidence, "
        "not in the Answer line.\n"
        "6. User-provided temporary context can clarify the current question or provide text to analyze, but it is not "
        "retrieved evidence. Do not cite it with context IDs.\n"
        "7. Add brief Evidence bullets with context IDs like [1] or [2]. Use conversation history only to resolve references; "
        "the Context is the source of RAG evidence.\n\n"
        "Preferred output format:\n"
        "Answer: <short answer only>\n"
        "Evidence: <optional brief supporting notes with context IDs>\n\n"
        f"Context:\n{context_str}\n\n"
        f"User-provided temporary context:\n{temporary_context_str}\n\n"
        f"Conversation history:\n{history_str}\n\n"
        f"Question: {question}"
    )
