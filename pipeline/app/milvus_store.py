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
        field_names = {field.name for field in collection.schema.fields}
        if "user_id" not in field_names:
            raise RuntimeError(
                f"Milvus collection '{name}' does not have user_id field. "
                "Create a new collection or rebuild it before enabling private knowledge isolation."
            )
        _chunk_field_name(collection)
        _vector_field_name(collection)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
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
    user_id: str,
    doc_id: str,
    chunk_count: int,
    embeddings: List[List[float]],
) -> List[int]:
    if chunk_count != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")

    chunk_field = _chunk_field_name(collection)
    vector_field = _vector_field_name(collection)
    field_names = {field.name for field in collection.schema.fields}

    rows: List[dict[str, Any]] = []
    for index, embedding in enumerate(embeddings):
        row: dict[str, Any] = {
            "user_id": user_id,
            "doc_id": doc_id,
            chunk_field: index,
            vector_field: embedding,
        }
        if "text" in field_names:
            row["text"] = ""
        rows.append(row)

    result = collection.insert(rows)
    return [int(primary_key) for primary_key in result.primary_keys]


def search(collection: Collection, user_id: str, query_vector: List[float], top_k: int) -> List[dict]:
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    chunk_field = _chunk_field_name(collection)
    vector_field = _vector_field_name(collection)
    result = collection.search(
        data=[query_vector],
        anns_field=vector_field,
        param=search_params,
        limit=top_k,
        expr=f'user_id == "{_escape_expr_value(user_id)}"',
        output_fields=["user_id", "doc_id", chunk_field],
    )

    hits = []
    for hit in result[0]:
        chunk_index = hit.entity.get(chunk_field)
        hits.append(
            {
                "id": int(hit.id),
                "score": float(hit.distance),
                "user_id": hit.entity.get("user_id"),
                "doc_id": hit.entity.get("doc_id"),
                "chunk_index": int(chunk_index) if chunk_index is not None else None,
            }
        )
    return hits


def _chunk_field_name(collection: Collection) -> str:
    field_names = {field.name for field in collection.schema.fields}
    if "chunk_index" in field_names:
        return "chunk_index"
    if "chunk_size" in field_names:
        return "chunk_size"
    raise RuntimeError(
        f"Milvus collection '{collection.name}' must have chunk_index or chunk_size INT64 field."
    )


def _vector_field_name(collection: Collection) -> str:
    for field in collection.schema.fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            return field.name
    raise RuntimeError(f"Milvus collection '{collection.name}' must have a FLOAT_VECTOR field.")


def _escape_expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
