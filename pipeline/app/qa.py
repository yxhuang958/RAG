from __future__ import annotations

import time

from .config import get_settings
from .llm import LLMClient
from .prompt import build_prompt
from .rerank import Reranker
from .retriever import Retriever
from .second_stage_ranker import SecondStageRanker


class QAPipeline:
    def __init__(self, user_id: str) -> None:
        settings = get_settings()
        self.rag_enabled = settings.rag_enabled
        self.rerank_enabled = settings.rerank_enabled
        self.second_stage_enabled = settings.second_stage_rerank_enabled
        self.context_top_k = settings.rerank_top_k
        self.retriever = Retriever(user_id) if self.rag_enabled else None
        self.reranker = Reranker() if self.rag_enabled and self.rerank_enabled else None
        self.second_stage_ranker = (
            SecondStageRanker() if self.rag_enabled and self.second_stage_enabled else None
        )
        self.llm = LLMClient()

    def answer(self, collection, messages: list[dict], top_k: int) -> dict:
        total_start = time.perf_counter()
        question = messages[-1]["content"]

        start = time.perf_counter()
        if self.retriever is None:
            hits = []
            reranked = []
            _log_timing("qa.retrieve_skipped", start, "rag_enabled=false")
            start = time.perf_counter()
            _log_timing("qa.rerank_skipped", start, "rag_enabled=false contexts=0")
        else:
            hits = self.retriever.retrieve(collection, question, top_k)
            _log_timing("qa.retrieve", start, f"contexts={len(hits)}")

            start = time.perf_counter()
            if self.reranker is None:
                reranked = hits[: self.context_top_k]
                _log_timing("qa.rerank_skipped", start, f"contexts={len(reranked)}")
            else:
                reranked = self.reranker.rerank(question, hits)
                _log_timing("qa.rerank", start, f"contexts={len(reranked)}")

        start = time.perf_counter()
        if self.second_stage_ranker is None:
            _log_timing("qa.second_stage_rerank_skipped", start, f"contexts={len(reranked)}")
        else:
            reranked = self.second_stage_ranker.rerank(messages, reranked)
            _log_timing("qa.second_stage_rerank", start, f"contexts={len(reranked)}")

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
