from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import get_settings


class SecondStageRanker:
    _model: Optional[SentenceTransformer] = None
    _model_name: str | None = None

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.second_stage_rerank_model
        self.weight = min(1.0, max(0.0, settings.second_stage_rerank_weight))
        self.history_messages = max(0, settings.second_stage_history_messages)

        if SecondStageRanker._model is None or SecondStageRanker._model_name != self.model_name:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading second-stage rerank model on device: {device}")
            SecondStageRanker._model = SentenceTransformer(self.model_name, device=device)
            SecondStageRanker._model_name = self.model_name

    def rerank(self, messages: list[dict], passages: list[dict]) -> list[dict]:
        if not passages or SecondStageRanker._model is None:
            return passages

        intent = self._build_intent(messages)
        texts = [str(passage.get("text") or "") for passage in passages]
        embeddings = SecondStageRanker._model.encode(
            [intent, *texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = embeddings[0]
        passage_vectors = embeddings[1:]
        semantic_scores = [float(np.dot(query_vector, vector)) for vector in passage_vectors]
        semantic_scores = [(score + 1.0) / 2.0 for score in semantic_scores]
        base_scores = _normalize([_base_score(passage) for passage in passages])

        rescored = []
        for passage, base, semantic in zip(passages, base_scores, semantic_scores):
            final_score = (1.0 - self.weight) * base + self.weight * semantic
            rescored.append(
                {
                    **passage,
                    "second_stage_score": final_score,
                    "second_stage_semantic_score": semantic,
                    "second_stage_base_score": base,
                }
            )

        rescored.sort(key=lambda passage: passage.get("second_stage_score", 0.0), reverse=True)
        return rescored

    def _build_intent(self, messages: list[dict]) -> str:
        if not messages:
            return ""

        question = str(messages[-1].get("content") or "").strip()
        history = messages[:-1][-self.history_messages :] if self.history_messages else []
        history_text = "\n".join(
            f"{str(item.get('role') or 'user')}: {str(item.get('content') or '').strip()}"
            for item in history
            if str(item.get("content") or "").strip()
        )
        if not history_text:
            return question
        return f"{history_text}\nuser: {question}"


def _base_score(passage: dict) -> float:
    for key in ("rerank_score", "rrf_score", "recall_score", "score"):
        value = passage.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        return [0.5 for _ in values]

    span = max_value - min_value
    return [(value - min_value) / span for value in values]
