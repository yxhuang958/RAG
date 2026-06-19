from __future__ import annotations

import threading
from typing import Any, List

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from .config import get_settings


_CONNECTION_ALIAS = "default"
_CONNECTION_LOCK = threading.Lock()


def connect_milvus() -> None:
    with _CONNECTION_LOCK:
        if connections.has_connection(_CONNECTION_ALIAS):
            return

        try:
            _connect_milvus()
        except Exception:
            _reset_milvus_connection()
            _connect_milvus()


def _connect_milvus() -> None:
    settings = get_settings()
    if settings.milvus_uri:
        connections.connect(alias=_CONNECTION_ALIAS, uri=settings.milvus_uri, token=settings.milvus_token)
        return

    connections.connect(
        alias=_CONNECTION_ALIAS,
        host=settings.milvus_host,
        port=settings.milvus_port,
        user=settings.milvus_user,
        password=settings.milvus_password,
    )


def _reset_milvus_connection() -> None:
    try:
        connections.disconnect(_CONNECTION_ALIAS)
    except Exception:
        pass
    try:
        connections.remove_connection(_CONNECTION_ALIAS)
    except Exception:
        pass


def get_or_create_collection(dim: int) -> Collection:
    try:
        return _get_or_create_collection(dim)
    except Exception:
        with _CONNECTION_LOCK:
            _reset_milvus_connection()
            _connect_milvus()
        return _get_or_create_collection(dim)


def _get_or_create_collection(dim: int) -> Collection:
    settings = get_settings()
    name = settings.milvus_collection

    if utility.has_collection(name):
        collection = Collection(name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="CANN RAG chunks")
        collection = Collection(name, schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index("embedding", index_params)

    collection.load()
    return collection


def insert_chunks(
    collection: Collection,
    doc_id: str,
    chunk_count: int,
    embeddings: List[List[float]],
) -> List[int]:
    if chunk_count != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")

    indices = list(range(chunk_count))
    doc_ids = [doc_id for _ in range(chunk_count)]
    field_names = {field.name for field in collection.schema.fields}
    if "text" in field_names:
        data: List[Any] = [doc_ids, indices, ["" for _ in range(chunk_count)], embeddings]
    else:
        data = [doc_ids, indices, embeddings]
    result = collection.insert(data)
    collection.flush()
    return [int(primary_key) for primary_key in result.primary_keys]


def search(collection: Collection, query_vector: List[float], top_k: int) -> List[dict]:
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    result = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["doc_id", "chunk_index"],
    )

    hits = []
    for hit in result[0]:
        chunk_index = hit.entity.get("chunk_index")
        hits.append(
            {
                "id": int(hit.id),
                "score": float(hit.distance),
                "doc_id": hit.entity.get("doc_id"),
                "chunk_index": int(chunk_index) if chunk_index is not None else None,
            }
        )
    return hits
