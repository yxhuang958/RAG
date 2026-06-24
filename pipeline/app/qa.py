from __future__ import annotations

import time

from .config import get_settings
from .llm import LLMClient
from .prompt import build_prompt
from .query_preprocessor import ProcessedQuery, preprocess_query
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
        self.context_max_per_doc = max(1, settings.context_max_per_doc)
        self.retriever = Retriever(user_id) if self.rag_enabled else None
        self.reranker = Reranker() if self.rag_enabled and self.rerank_enabled else None
        self.second_stage_ranker = (
            SecondStageRanker() if self.rag_enabled and self.second_stage_enabled else None
        )
        self.query_preprocess_enabled = settings.query_preprocess_enabled
        self.query_long_threshold_chars = settings.query_long_threshold_chars
        self.query_context_retrieval_chars = settings.query_context_retrieval_chars
        self.query_temp_context_prompt_chars = settings.query_temp_context_prompt_chars
        self.llm = LLMClient()

    def answer(self, collection, messages: list[dict], top_k: int) -> dict:
        total_start = time.perf_counter()
        processed_query = self._preprocess_query(messages[-1]["content"])
        retrieval_query = processed_query.retrieval_query

        start = time.perf_counter()
        if self.retriever is None:
            hits = []
            reranked = []
            _log_timing("qa.retrieve_skipped", start, "rag_enabled=false")
            start = time.perf_counter()
            _log_timing("qa.rerank_skipped", start, "rag_enabled=false contexts=0")
        else:
            hits = self.retriever.retrieve(collection, retrieval_query, top_k)
            _log_timing("qa.retrieve", start, f"contexts={len(hits)}")

            start = time.perf_counter()
            if self.reranker is None:
                reranked = hits[: self.context_top_k]
                _log_timing("qa.rerank_skipped", start, f"contexts={len(reranked)}")
            else:
                reranked = self.reranker.rerank(retrieval_query, hits)
                _log_timing("qa.rerank", start, f"contexts={len(reranked)}")

        start = time.perf_counter()
        if self.second_stage_ranker is None:
            _log_timing("qa.second_stage_rerank_skipped", start, f"contexts={len(reranked)}")
        else:
            reranked = self.second_stage_ranker.rerank(messages, reranked, intent=retrieval_query)
            _log_timing("qa.second_stage_rerank", start, f"contexts={len(reranked)}")

        start = time.perf_counter()
        reranked = _finalize_contexts(reranked, self.context_top_k, self.context_max_per_doc)
        _log_timing("qa.context_finalize", start, f"contexts={len(reranked)}")

        start = time.perf_counter()
        prompt = build_prompt(messages, reranked, processed_query)
        _log_timing("qa.prompt_build", start, f"prompt_chars={len(prompt)}")

        start = time.perf_counter()
        answer = self.llm.generate(prompt)
        _log_timing("qa.llm", start, f"answer_chars={len(answer)}")
        _log_timing("qa.total", total_start)
        return {
            "answer": answer,
            "contexts": reranked,
            "query_analysis": {
                "question": processed_query.question,
                "retrieval_query": processed_query.retrieval_query,
                "temporary_context_chars": len(processed_query.temporary_context),
                "was_split": processed_query.was_split,
                "strategy": processed_query.strategy,
            },
        }

    def _preprocess_query(self, question: str) -> ProcessedQuery:
        if not self.query_preprocess_enabled:
            return ProcessedQuery(
                original=question,
                question=question,
                temporary_context="",
                retrieval_query=question,
                was_split=False,
                strategy="disabled",
            )
        return preprocess_query(
            question,
            long_threshold_chars=self.query_long_threshold_chars,
            retrieval_context_chars=self.query_context_retrieval_chars,
            prompt_context_chars=self.query_temp_context_prompt_chars,
        )


def _log_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    # print(f"TIMING query {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)


def _finalize_contexts(contexts: list[dict], limit: int, max_per_doc: int = 3) -> list[dict]:
    unique_contexts: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for context in contexts:
        doc_id = context.get("doc_id")
        source_chunk_index = context.get("source_chunk_index", context.get("chunk_index"))
        if not isinstance(doc_id, str) or not isinstance(source_chunk_index, int):
            unique_contexts.append(context)
            continue

        key = (doc_id, source_chunk_index)
        if key in seen:
            continue
        seen.add(key)

        expanded_text = str(context.get("expanded_text") or "").strip()
        if not expanded_text:
            unique_contexts.append(context)
            continue

        unique_contexts.append(
            {
                **context,
                "text": expanded_text,
                "evidence_text": context.get("text"),
                "chunk_index": source_chunk_index,
            }
        )

    return _select_diverse_contexts(unique_contexts, max(1, limit), max(1, max_per_doc))


def _select_diverse_contexts(contexts: list[dict], limit: int, max_per_doc: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    doc_order: list[str] = []

    for index, context in enumerate(contexts):
        doc_id = str(context.get("doc_id") or f"_missing_doc_{index}")
        if doc_id not in grouped:
            grouped[doc_id] = []
            doc_order.append(doc_id)
        grouped[doc_id].append(context)

    selected: list[dict] = []
    doc_counts = {doc_id: 0 for doc_id in doc_order}

    for round_index in range(max_per_doc):
        made_progress = False
        for doc_id in doc_order:
            if doc_counts[doc_id] > round_index:
                continue
            bucket = grouped[doc_id]
            if not bucket:
                continue

            selected.append(bucket.pop(0))
            doc_counts[doc_id] += 1
            made_progress = True
            if len(selected) >= limit:
                return selected

        if not made_progress:
            break

    for doc_id in doc_order:
        while grouped[doc_id] and len(selected) < limit:
            selected.append(grouped[doc_id].pop(0))
        if len(selected) >= limit:
            break

    return selected
