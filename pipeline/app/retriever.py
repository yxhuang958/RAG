from __future__ import annotations

import time
from typing import List

from .config import get_settings
from .document_store import DocumentStore
from .embedding import EmbeddingClient
from .milvus_store import search


class Retriever:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.embedding = EmbeddingClient()
        self.document_store = DocumentStore(get_settings().document_store_path)

    def retrieve(self, collection, query: str, top_k: int) -> List[dict]:
        total_start = time.perf_counter()

        start = time.perf_counter()
        vector = self.embedding.embed([query])[0]
        _log_timing("retriever.embedding", start)

        start = time.perf_counter()
        hits = search(collection, self.user_id, vector, top_k)
        _log_timing("retriever.milvus_search", start, f"hits={len(hits)}")

        start = time.perf_counter()
        hydrated = self.document_store.hydrate_hits(self.user_id, hits)
        _log_timing("retriever.hydrate_chunks", start, f"contexts={len(hydrated)}")
        _log_timing("retriever.total", total_start)
        return hydrated


def _log_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    print(f"TIMING query {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)
