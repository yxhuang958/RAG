from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import get_settings

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
except ImportError:  # pragma: no cover - optional unless Mongo document store is enabled
    ASCENDING = DESCENDING = MongoClient = UpdateOne = None


class DocumentStore:
    def __init__(self, path: str) -> None:
        settings = get_settings()
        self.path = Path(path)
        self._lock = threading.Lock()
        self.backend = settings.document_store_backend.strip().lower()
        self._mongo_client = None
        self._documents = None
        self._chunks = None

        if self.backend == "mongodb":
            if MongoClient is None:
                raise RuntimeError("pymongo is required when DOCUMENT_STORE_BACKEND=mongodb")
            self._mongo_client = MongoClient(
                settings.mongo_uri,
                serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
                connectTimeoutMS=settings.mongo_timeout_ms,
                socketTimeoutMS=settings.mongo_timeout_ms,
                maxPoolSize=settings.mongo_max_pool_size,
            )
            self._mongo_client.admin.command("ping")
            db = self._mongo_client[settings.mongo_db]
            self._documents = db[settings.mongo_documents_collection]
            self._chunks = db[settings.mongo_chunks_collection]
            self._ensure_mongo_indexes()

    def get(self, content_hash: str) -> dict[str, Any] | None:
        if self._documents is not None:
            item = self._documents.find_one({"content_hash": content_hash}, {"_id": 0})
            return dict(item) if item else None

        with self._lock:
            documents = self._read()
            item = documents.get(content_hash)
            return dict(item) if item else None

    def reserve(
        self,
        *,
        user_id: str,
        content_hash: str,
        file_hash: str,
        doc_id: str,
        filename: str,
        chunks: int,
    ) -> tuple[dict[str, Any], bool]:
        now = int(time.time())
        key = self._key(user_id, content_hash)
        if self._documents is not None:
            existing = self._documents.find_one({"key": key}, {"_id": 0})
            if existing and existing.get("status") in {"ready", "processing"}:
                return dict(existing), False

            item = {
                "key": key,
                "user_id": user_id,
                "content_hash": content_hash,
                "file_hash": file_hash,
                "doc_id": doc_id,
                "filename": filename,
                "chunks": chunks,
                "status": "processing",
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            self._documents.update_one({"key": key}, {"$set": item}, upsert=True)
            return dict(item), True

        with self._lock:
            documents = self._read()
            existing = documents.get(key)
            if existing and existing.get("status") in {"ready", "processing"}:
                return dict(existing), False

            item = {
                "user_id": user_id,
                "content_hash": content_hash,
                "file_hash": file_hash,
                "doc_id": doc_id,
                "filename": filename,
                "chunks": chunks,
                "status": "processing",
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            documents[key] = item
            self._write(documents)
            return dict(item), True

    def mark_ready(
        self,
        content_hash: str,
        *,
        user_id: str,
        collection: str,
        ids: list[int],
        chunk_texts: list[str],
    ) -> dict[str, Any]:
        now = int(time.time())
        key = self._key(user_id, content_hash)
        if self._documents is not None:
            item = self._documents.find_one({"key": key}, {"_id": 0})
            if not item:
                raise KeyError(key)

            item["status"] = "ready"
            item["collection"] = collection
            item["ids"] = ids
            item["chunks"] = len(chunk_texts)
            item["updated_at"] = now
            self._documents.update_one({"key": key}, {"$set": item}, upsert=True)
            self._replace_chunks(
                user_id=user_id,
                doc_id=str(item["doc_id"]),
                filename=item.get("filename"),
                content_hash=content_hash,
                chunk_texts=chunk_texts,
                now=now,
            )
            return dict(item)

        with self._lock:
            documents = self._read()
            item = documents[key]
            item["status"] = "ready"
            item["collection"] = collection
            item["ids"] = ids
            item["chunk_texts"] = chunk_texts
            item["updated_at"] = now
            documents[key] = item
            self._write(documents)
            return dict(item)

    def mark_failed(self, user_id: str, content_hash: str, error: str) -> None:
        now = int(time.time())
        key = self._key(user_id, content_hash)
        if self._documents is not None:
            self._documents.update_one(
                {"key": key},
                {"$set": {"status": "failed", "error": error, "updated_at": now}},
            )
            return

        with self._lock:
            documents = self._read()
            item = documents.get(key)
            if not item:
                return
            item["status"] = "failed"
            item["error"] = error
            item["updated_at"] = now
            documents[key] = item
            self._write(documents)

    def attach_chunk_texts(self, user_id: str, content_hash: str, chunk_texts: list[str]) -> dict[str, Any] | None:
        now = int(time.time())
        key = self._key(user_id, content_hash)
        if self._documents is not None:
            item = self._documents.find_one({"key": key}, {"_id": 0})
            if not item:
                return None

            item["chunks"] = len(chunk_texts)
            item["updated_at"] = now
            self._documents.update_one(
                {"key": key},
                {"$set": {"chunks": len(chunk_texts), "updated_at": now}},
            )
            self._replace_chunks(
                user_id=user_id,
                doc_id=str(item["doc_id"]),
                filename=item.get("filename"),
                content_hash=content_hash,
                chunk_texts=chunk_texts,
                now=now,
            )
            return dict(item)

        with self._lock:
            documents = self._read()
            item = documents.get(key)
            if not item:
                return None

            item["chunk_texts"] = chunk_texts
            item["chunks"] = len(chunk_texts)
            item["updated_at"] = now
            documents[key] = item
            self._write(documents)
            return dict(item)

    def get_chunk(self, user_id: str, doc_id: str, chunk_index: int) -> str | None:
        if self._chunks is not None:
            item = self._chunks.find_one(
                {"user_id": user_id, "doc_id": doc_id, "chunk_index": chunk_index},
                {"_id": 0, "text": 1},
            )
            text = item.get("text") if item else None
            return text if isinstance(text, str) else None

        with self._lock:
            documents = self._read()
            for item in documents.values():
                if (
                    item.get("user_id") != user_id
                    or item.get("doc_id") != doc_id
                    or item.get("status") != "ready"
                ):
                    continue

                chunk_texts = item.get("chunk_texts") or []
                if 0 <= chunk_index < len(chunk_texts):
                    text = chunk_texts[chunk_index]
                    return text if isinstance(text, str) else None
        return None

    def get_chunk_window(
        self,
        user_id: str,
        doc_id: str,
        chunk_index: int,
        radius: int,
    ) -> tuple[str, list[int]] | None:
        radius = max(0, radius)
        if self._chunks is not None:
            start = max(0, chunk_index - radius)
            end = chunk_index + radius
            cursor = self._chunks.find(
                {
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_index": {"$gte": start, "$lte": end},
                },
                {"_id": 0, "chunk_index": 1, "text": 1},
            ).sort("chunk_index", ASCENDING)
            indices: list[int] = []
            texts: list[str] = []
            for item in cursor:
                text = item.get("text")
                index = item.get("chunk_index")
                if not isinstance(text, str) or not text.strip() or not isinstance(index, int):
                    continue
                indices.append(index)
                texts.append(text.strip())

            if not texts:
                return None
            return "\n\n".join(texts), indices

        with self._lock:
            documents = self._read()
            for item in documents.values():
                if (
                    item.get("user_id") != user_id
                    or item.get("doc_id") != doc_id
                    or item.get("status") != "ready"
                ):
                    continue

                chunk_texts = item.get("chunk_texts") or []
                if not 0 <= chunk_index < len(chunk_texts):
                    return None

                start = max(0, chunk_index - radius)
                end = min(len(chunk_texts), chunk_index + radius + 1)
                indices: list[int] = []
                texts: list[str] = []
                for index in range(start, end):
                    text = chunk_texts[index]
                    if not isinstance(text, str) or not text.strip():
                        continue
                    indices.append(index)
                    texts.append(text.strip())

                if not texts:
                    return None
                return "\n\n".join(texts), indices
        return None

    def hydrate_hits(
        self,
        user_id: str,
        hits: list[dict[str, Any]],
        neighbor_window: int = 0,
    ) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for hit in hits:
            doc_id = hit.get("doc_id")
            chunk_index = hit.get("chunk_index")
            if not isinstance(doc_id, str) or not isinstance(chunk_index, int):
                continue

            window = self.get_chunk_window(user_id, doc_id, chunk_index, neighbor_window)
            if not window:
                continue
            text, source_indices = window

            hydrated.append(
                {
                    **hit,
                    "text": text,
                    "source_chunk_index": chunk_index,
                    "source_chunk_indices": source_indices,
                }
            )
        return hydrated

    def list_user_chunks(self, user_id: str) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        if self._chunks is not None:
            cursor = self._chunks.find(
                {"user_id": user_id},
                {"_id": 0, "user_id": 1, "doc_id": 1, "filename": 1, "chunk_index": 1, "text": 1},
            ).sort([("doc_id", ASCENDING), ("chunk_index", ASCENDING)])
            for item in cursor:
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(dict(item))
            return chunks

        with self._lock:
            documents = self._read()
            for item in documents.values():
                if item.get("user_id") != user_id or item.get("status") != "ready":
                    continue

                doc_id = item.get("doc_id")
                if not isinstance(doc_id, str):
                    continue

                filename = item.get("filename")
                chunk_texts = item.get("chunk_texts") or []
                for chunk_index, text in enumerate(chunk_texts):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    chunks.append(
                        {
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "filename": filename,
                            "chunk_index": chunk_index,
                            "text": text,
                        }
                    )
        return chunks

    def _key(self, user_id: str, content_hash: str) -> str:
        return f"{user_id}:{content_hash}"

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        content = self.path.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        data = json.loads(content)
        return data if isinstance(data, dict) else {}

    def _write(self, documents: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def _ensure_mongo_indexes(self) -> None:
        if self._documents is None or self._chunks is None:
            return
        self._documents.create_index([("key", ASCENDING)], unique=True, name="key_unique")
        self._documents.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)], name="user_updated_at")
        self._documents.create_index([("doc_id", ASCENDING)], name="doc_id")
        self._chunks.create_index(
            [("user_id", ASCENDING), ("doc_id", ASCENDING), ("chunk_index", ASCENDING)],
            unique=True,
            name="user_doc_chunk_unique",
        )
        self._chunks.create_index([("user_id", ASCENDING)], name="user_id")
        self._chunks.create_index([("doc_id", ASCENDING), ("chunk_index", ASCENDING)], name="doc_chunk")

    def _replace_chunks(
        self,
        *,
        user_id: str,
        doc_id: str,
        filename: str | None,
        content_hash: str,
        chunk_texts: list[str],
        now: int,
    ) -> None:
        if self._chunks is None:
            return
        self._chunks.delete_many({"user_id": user_id, "doc_id": doc_id})
        if not chunk_texts:
            return
        operations = [
            UpdateOne(
                {"user_id": user_id, "doc_id": doc_id, "chunk_index": index},
                {
                    "$set": {
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "content_hash": content_hash,
                        "chunk_index": index,
                        "text": text,
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            for index, text in enumerate(chunk_texts)
        ]
        self._chunks.bulk_write(operations, ordered=False)
