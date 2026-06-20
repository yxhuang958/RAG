from __future__ import annotations

import threading
import time

from pymilvus import Collection

from .config import get_settings
from .milvus_store import connect_milvus, get_or_create_collection
from .qa import QAPipeline


_COLLECTION_LOCK = threading.Lock()
_COLLECTION: Collection | None = None

_PIPELINE_LOCK = threading.Lock()
_PIPELINES: dict[str, QAPipeline] = {}


def get_collection(dim: int) -> Collection:
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    with _COLLECTION_LOCK:
        if _COLLECTION is None:
            start = time.perf_counter()
            connect_milvus()
            _COLLECTION = get_or_create_collection(dim)
            _log_warmup_timing("milvus_collection", start)
        return _COLLECTION


def get_pipeline(user_id: str) -> QAPipeline:
    with _PIPELINE_LOCK:
        pipeline = _PIPELINES.get(user_id)
        if pipeline is None:
            start = time.perf_counter()
            pipeline = QAPipeline(user_id)
            _PIPELINES[user_id] = pipeline
            _log_warmup_timing("qa_pipeline", start, f"user_id={user_id}")
        return pipeline


def warmup(user_id: str) -> dict:
    settings = get_settings()
    total_start = time.perf_counter()
    get_collection(settings.embedding_dim)
    get_pipeline(user_id)
    elapsed_ms = (time.perf_counter() - total_start) * 1000
    print(f"TIMING warmup total={elapsed_ms:.2f}ms user_id={user_id}", flush=True)
    return {
        "status": "ok",
        "user_id": user_id,
        "collection": settings.milvus_collection,
        "elapsed_ms": elapsed_ms,
    }


def _log_warmup_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    print(f"TIMING warmup {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)
