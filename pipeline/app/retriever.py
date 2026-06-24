from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Iterable, List

from .config import get_settings
from .document_store import DocumentStore
from .embedding import EmbeddingClient
from .milvus_store import search


class Retriever:
    def __init__(self, user_id: str) -> None:
        settings = get_settings()
        self.user_id = user_id
        self.multi_retrieval_enabled = settings.multi_retrieval_enabled
        self.dense_factor = max(1, settings.multi_retrieval_dense_factor)
        self.lexical_factor = max(1, settings.multi_retrieval_lexical_factor)
        self.rrf_k = max(1, settings.multi_retrieval_rrf_k)
        self.neighbor_window = max(0, settings.retrieval_neighbor_window)
        self.evidence_window_sentences = max(0, settings.evidence_window_sentences)
        self.evidence_candidate_factor = max(1, settings.evidence_candidate_factor)
        self.embedding = EmbeddingClient()
        self.document_store = DocumentStore(settings.document_store_path)

    def retrieve(self, collection, query: str, top_k: int) -> List[dict]:
        total_start = time.perf_counter()
        requested_top_k = max(1, top_k)
        candidate_limit = requested_top_k * self.evidence_candidate_factor
        dense_limit = requested_top_k * self.dense_factor if self.multi_retrieval_enabled else candidate_limit

        start = time.perf_counter()
        vector = self.embedding.embed([query])[0]
        _log_timing("retriever.embedding", start)

        start = time.perf_counter()
        hits = search(collection, self.user_id, vector, dense_limit)
        _log_timing("retriever.milvus_search", start, f"hits={len(hits)}")

        start = time.perf_counter()
        dense_hits = self.document_store.hydrate_hits(self.user_id, hits, self.neighbor_window)
        _log_timing("retriever.hydrate_chunks", start, f"contexts={len(dense_hits)}")

        if not self.multi_retrieval_enabled:
            start = time.perf_counter()
            evidence_hits = self._evidence_candidates(query, dense_hits, candidate_limit)
            _log_timing("retriever.evidence_candidates", start, f"contexts={len(evidence_hits)}")
            _log_timing("retriever.total", total_start, "mode=dense")
            return evidence_hits

        start = time.perf_counter()
        lexical_hits = self._lexical_search(query, requested_top_k * self.lexical_factor)
        _log_timing("retriever.lexical_search", start, f"hits={len(lexical_hits)}")

        start = time.perf_counter()
        lexical_hits = self.document_store.hydrate_hits(self.user_id, lexical_hits, self.neighbor_window)
        _log_timing("retriever.hydrate_lexical_chunks", start, f"contexts={len(lexical_hits)}")

        start = time.perf_counter()
        fused = self._rrf_fuse([dense_hits, lexical_hits], candidate_limit)
        _log_timing("retriever.fuse", start, f"contexts={len(fused)}")

        start = time.perf_counter()
        evidence_hits = self._evidence_candidates(query, fused, candidate_limit)
        _log_timing("retriever.evidence_candidates", start, f"contexts={len(evidence_hits)}")
        _log_timing("retriever.total", total_start)
        return evidence_hits

    def _lexical_search(self, query: str, limit: int) -> List[dict]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        chunks = self.document_store.list_user_chunks(self.user_id)
        if not chunks:
            return []

        doc_term_counts: list[Counter[str]] = []
        doc_freq: Counter[str] = Counter()
        lengths: list[int] = []
        for chunk in chunks:
            terms = _tokenize(chunk["text"])
            counts = Counter(terms)
            doc_term_counts.append(counts)
            lengths.append(sum(counts.values()))
            doc_freq.update(counts.keys())

        avg_len = sum(lengths) / len(lengths) if lengths else 1.0
        scored: list[dict] = []
        for chunk, counts, length in zip(chunks, doc_term_counts, lengths):
            score = _bm25_score(query_counts, counts, doc_freq, len(chunks), length, avg_len)
            if score <= 0:
                continue
            scored.append({**chunk, "lexical_score": score, "score": score})

        scored.sort(key=lambda item: item["lexical_score"], reverse=True)
        return scored[:limit]

    def _rrf_fuse(self, routes: Iterable[List[dict]], top_k: int) -> List[dict]:
        fused: dict[tuple[str, int], dict] = {}
        for route_name, hits in zip(("dense", "lexical"), routes):
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit.get("doc_id")
                chunk_index = hit.get("chunk_index")
                if not isinstance(doc_id, str) or not isinstance(chunk_index, int):
                    continue

                key = (doc_id, chunk_index)
                item = fused.setdefault(key, {**hit, "routes": [], "rrf_score": 0.0})
                item["rrf_score"] += 1.0 / (self.rrf_k + rank)
                item["routes"].append(route_name)
                if route_name == "dense" and "recall_score" in hit:
                    item["recall_score"] = hit["recall_score"]
                if route_name == "lexical" and "lexical_score" in hit:
                    item["lexical_score"] = hit["lexical_score"]

        results = list(fused.values())
        results.sort(key=lambda item: item["rrf_score"], reverse=True)
        for item in results:
            item["score"] = item["rrf_score"]
        return results[:top_k]

    def _evidence_candidates(self, query: str, hits: List[dict], limit: int) -> List[dict]:
        candidates: list[dict] = []
        seen: set[tuple[str, int, str]] = set()
        max_per_hit = max(2, self.evidence_window_sentences + 2)

        for hit in hits:
            text = str(hit.get("text") or "").strip()
            doc_id = hit.get("doc_id")
            chunk_index = hit.get("chunk_index")
            if not text or not isinstance(doc_id, str) or not isinstance(chunk_index, int):
                continue

            snippets = _sentence_windows(text, self.evidence_window_sentences)
            scored_snippets = sorted(
                (
                    (_overlap_score(query, snippet), index, snippet)
                    for index, snippet in enumerate(snippets)
                    if snippet.strip()
                ),
                key=lambda item: item[0],
                reverse=True,
            )

            for evidence_score, evidence_index, snippet in scored_snippets[:max_per_hit]:
                key = (doc_id, chunk_index, _normalize_snippet(snippet))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        **hit,
                        "text": snippet,
                        "evidence_index": evidence_index,
                        "evidence_score": evidence_score,
                        "expanded_text": text,
                    }
                )

        candidates.sort(
            key=lambda item: (
                float(item.get("evidence_score", 0.0)),
                _base_score(item),
            ),
            reverse=True,
        )
        return candidates[:limit]


_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SENTENCE_RE = re.compile(r"[^.!?;。！？；]+[.!?;。！？；]?")


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    latin_tokens = _LATIN_TOKEN_RE.findall(normalized)
    cjk_chars = _CJK_CHAR_RE.findall(normalized)
    cjk_bigrams = [left + right for left, right in zip(cjk_chars, cjk_chars[1:])]
    return latin_tokens + cjk_chars + cjk_bigrams


def _overlap_score(query: str, passage: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    passage_tokens = set(_tokenize(passage))
    return len(query_tokens.intersection(passage_tokens)) / math.sqrt(len(query_tokens))


def _sentence_windows(text: str, radius: int) -> list[str]:
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
    if not sentences:
        return [text]
    if len(sentences) <= radius * 2 + 1:
        return [" ".join(sentences)]

    windows: list[str] = []
    for index in range(len(sentences)):
        start = max(0, index - radius)
        end = min(len(sentences), index + radius + 1)
        windows.append(" ".join(sentences[start:end]))
    return windows


def _normalize_snippet(text: str) -> str:
    return " ".join(text.lower().split())


def _base_score(hit: dict) -> float:
    for key in ("rrf_score", "recall_score", "lexical_score", "score"):
        value = hit.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _bm25_score(
    query_counts: Counter[str],
    doc_counts: Counter[str],
    doc_freq: Counter[str],
    doc_count: int,
    doc_len: int,
    avg_doc_len: float,
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term, query_weight in query_counts.items():
        tf = doc_counts.get(term, 0)
        if tf <= 0:
            continue

        df = doc_freq.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        norm = tf + k1 * (1 - b + b * doc_len / (avg_doc_len or 1.0))
        score += query_weight * idf * (tf * (k1 + 1)) / norm
    return float(score)


def _log_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    # print(f"TIMING query {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)
