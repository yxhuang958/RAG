from __future__ import annotations

from typing import List

from .config import get_settings
from .document_store import DocumentStore
from .embedding import EmbeddingClient
from .milvus_store import search


class Retriever:
    def __init__(self) -> None:
        self.embedding = EmbeddingClient()
        self.document_store = DocumentStore(get_settings().document_store_path)

    def retrieve(self, collection, query: str, top_k: int) -> List[dict]:
        vector = self.embedding.embed([query])[0]
        hits = search(collection, vector, top_k)
        return self.document_store.hydrate_hits(hits)
