from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class DocumentStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def get(self, content_hash: str) -> dict[str, Any] | None:
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

    def hydrate_hits(self, user_id: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for hit in hits:
            doc_id = hit.get("doc_id")
            chunk_index = hit.get("chunk_index")
            if not isinstance(doc_id, str) or not isinstance(chunk_index, int):
                continue

            text = self.get_chunk(user_id, doc_id, chunk_index)
            if not text:
                continue

            hydrated.append({**hit, "text": text})
        return hydrated

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
