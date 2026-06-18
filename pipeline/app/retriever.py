from __future__ import annotations

from typing import List

from .embedding import EmbeddingClient
from .milvus_store import search


class Retriever:
    def __init__(self) -> None:
        self.embedding = EmbeddingClient()

    def retrieve(self, collection, query: str, top_k: int) -> List[dict]:
        vector = self.embedding.embed([query])[0]
        return search(collection, vector, top_k)
