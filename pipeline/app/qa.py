from __future__ import annotations

import time

from .llm import LLMClient
from .prompt import build_prompt
from .rerank import Reranker
from .retriever import Retriever


class QAPipeline:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLMClient()

    def answer(self, collection, messages: list[dict], top_k: int) -> dict:
        total_start = time.perf_counter()
        question = messages[-1]["content"]

        start = time.perf_counter()
        hits = self.retriever.retrieve(collection, question, top_k)
        _log_timing("qa.retrieve", start, f"contexts={len(hits)}")

        start = time.perf_counter()
        reranked = self.reranker.rerank(question, hits)
        _log_timing("qa.rerank", start, f"contexts={len(reranked)}")

        start = time.perf_counter()
        prompt = build_prompt(messages, reranked)
        _log_timing("qa.prompt_build", start, f"prompt_chars={len(prompt)}")

        start = time.perf_counter()
        answer = self.llm.generate(prompt)
        _log_timing("qa.llm", start, f"answer_chars={len(answer)}")
        _log_timing("qa.total", total_start)
        return {
            "answer": answer,
            "contexts": reranked,
        }


def _log_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    print(f"TIMING query {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)
