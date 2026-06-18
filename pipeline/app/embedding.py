from __future__ import annotations

import hashlib
from typing import List, Optional

import httpx
# 确保导入了库
from sentence_transformers import SentenceTransformer

from .config import get_settings


def _mock_embed(text: str, dim: int) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255.0 for b in digest]
    if dim <= len(values):
        return values[:dim]
    repeats = (dim + len(values) - 1) // len(values)
    expanded = (values * repeats)[:dim]
    return expanded


class EmbeddingClient:
    _instance: Optional[SentenceTransformer] = None # 单例模式

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.embedding_provider
        self.endpoint = settings.embedding_endpoint
        self.api_key = settings.embedding_api_key
        self.model_name = settings.embedding_model
        self.dim = settings.embedding_dim

        # 如果是本地模式且尚未加载模型
        if self.provider == "local-st" and EmbeddingClient._instance is None:
            # 强制检测并使用可用设备，如果显卡驱动有问题则降级到 CPU
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading embedding model on device: {device}")
            EmbeddingClient._instance = SentenceTransformer(self.model_name, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # 本地推理逻辑
        if self.provider == "local-st" and EmbeddingClient._instance:
            embeddings = EmbeddingClient._instance.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        if self.provider == "mock" or not self.endpoint:
            return [_mock_embed(text, self.dim) for text in texts]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model_name, "texts": texts}
        with httpx.Client(timeout=30) as client:
            response = client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        vectors = data.get("vectors") or data.get("embeddings")
        if not vectors:
            raise ValueError("Embedding service returned empty vectors")
        return vectors
