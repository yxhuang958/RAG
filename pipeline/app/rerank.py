from __future__ import annotations

import math
import re
import time
from typing import Any, List, Optional
import httpx

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

# 新增 Google Cloud 导入
try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None

from .config import get_settings
from .sanitize import redact_secrets


_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    normalized = text.lower()
    latin_tokens = _LATIN_TOKEN_RE.findall(normalized)
    cjk_chars = _CJK_CHAR_RE.findall(normalized)
    cjk_bigrams = [left + right for left, right in zip(cjk_chars, cjk_chars[1:])]
    return latin_tokens + cjk_chars + cjk_bigrams


def _overlap_score(query: str, passage: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    passage_tokens = set(_tokenize(passage))
    overlap = query_tokens.intersection(passage_tokens)
    return len(overlap) / math.sqrt(len(query_tokens))


class Reranker:
    _instance: Optional[FlagReranker] = None
    _vertex_failures: int = 0
    _vertex_circuit_open_until: float = 0.0

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.rerank_provider
        self.endpoint = settings.rerank_endpoint
        self.api_key = settings.rerank_api_key
        self.model_name = settings.rerank_model
        self.top_k = settings.rerank_top_k
        self.weight = settings.rerank_weight
        self.circuit_breaker_seconds = settings.rerank_circuit_breaker_seconds
        self.circuit_breaker_failures = settings.rerank_circuit_breaker_failures
        self.vertex_retries = max(1, settings.rerank_vertex_retries)

        # GCP 相关配置
        self.project_id = settings.gcp_project_id
        self.location = settings.gcp_location
        self.vertex_endpoint_id = settings.vertex_rerank_endpoint_id

        # 调试信息：检查初始化状态
        print(f"DEBUG - Provider: {self.provider}")
        print(f"DEBUG - GCP Project: {self.project_id}")
        print(f"DEBUG - Vertex Endpoint ID: {self.vertex_endpoint_id}")
        print(f"DEBUG - AI Platform Lib Loaded: {aiplatform is not None}")

        # 初始化 Vertex AI
        if self.provider == "vertex" and aiplatform and self.project_id:
            print(f"DEBUG - Initializing Vertex AI in {self.location}")
            aiplatform.init(project=self.project_id, location=self.location)

        # 如果是本地模式且尚未加载模型
        if self.provider == "local-bge" and FlagReranker and Reranker._instance is None:
            Reranker._instance = FlagReranker(self.model_name, use_fp16=True)

    def _is_transient_vertex_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return (
            "429" in message
            or "not yet ready" in message
            or "scale-up from zero" in message
            or "resource exhausted" in message
            or "temporarily unavailable" in message
        )

    def _is_vertex_circuit_open(self) -> bool:
        return time.time() < Reranker._vertex_circuit_open_until

    def _record_vertex_success(self) -> None:
        Reranker._vertex_failures = 0
        Reranker._vertex_circuit_open_until = 0.0

    def _record_vertex_failure(self, error: Exception) -> None:
        Reranker._vertex_failures += 1
        if Reranker._vertex_failures >= self.circuit_breaker_failures:
            Reranker._vertex_circuit_open_until = time.time() + self.circuit_breaker_seconds
            print(
                f"WARNING - Vertex rerank circuit opened for {self.circuit_breaker_seconds}s "
                f"after {Reranker._vertex_failures} failure(s): {redact_secrets(error)}",
                flush=True,
            )

    def _predict_vertex_with_retry(self, endpoint, instances: list[dict]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.vertex_retries):
            try:
                return endpoint.predict(instances=instances)
            except Exception as error:
                last_error = error
                if not self._is_transient_vertex_error(error) or attempt == self.vertex_retries - 1:
                    raise

                wait_seconds = 2 ** attempt
                print(
                    f"WARNING - Vertex rerank endpoint is warming up; "
                    f"retrying in {wait_seconds}s ({attempt + 1}/{self.vertex_retries}): "
                    f"{redact_secrets(error)}"
                )
                time.sleep(wait_seconds)

        raise last_error or RuntimeError("Vertex rerank prediction failed")

    def _call_model_rerank(self, query: str, passages: List[str]) -> List[float]:
        """调用重排模型"""
        if self.provider == "vertex" and aiplatform and self.vertex_endpoint_id:
            if self._is_vertex_circuit_open():
                remaining = max(0.0, Reranker._vertex_circuit_open_until - time.time())
                print(
                    f"WARNING - [Rerank Status]: CIRCUIT_OPEN (熔断中); "
                    f"using lexical fallback for {remaining:.1f}s",
                    flush=True,
                )
                return self._generic_rerank_logic(query, passages)

            endpoint = aiplatform.Endpoint(self.vertex_endpoint_id)

            candidate_payloads = [
                [{"inputs": [[query, p] for p in passages]}],
                [{"inputs": [query, p]} for p in passages],
                [{"inputs": {"query": query, "passage": p}} for p in passages],
                [{"inputs": {"source_sentence": query, "sentences": passages}}],
                [{"text": [query, p]} for p in passages],
                [{"query": query, "passage": p} for p in passages],
            ]

            response = None
            last_error = None
            for idx, instances in enumerate(candidate_payloads, start=1):
                # print(f"DEBUG - Attempting format {idx}: {instances[:1]}")
                try:
                    response = self._predict_vertex_with_retry(endpoint, instances)
                    break
                except Exception as e:
                    last_error = e
                    # print(f"DEBUG - Format {idx} failed: {e}")
                    if self._is_transient_vertex_error(e):
                        break

            if response is None:
                if last_error:
                    self._record_vertex_failure(last_error)
                print(
                    "WARNING - [Rerank Status]: FALLBACK (调用失败); using lexical fallback: "
                    f"{redact_secrets(last_error)}"
                )
                return self._generic_rerank_logic(query, passages)
            
            # print(f"DEBUG - Vertex AI Raw Predictions: {response.predictions}")
            
            # 通用分数提取逻辑
            scores = self._extract_vertex_scores(response.predictions)
            if len(scores) != len(passages):
                raise ValueError(
                    f"Vertex returned {len(scores)} scores for {len(passages)} passages: "
                    f"{response.predictions}"
                )
            
            print(f"INFO - [Rerank Status]: SUCCESS (语义重排成功), Provider: {self.provider}")
            self._record_vertex_success()
            return scores

        if self.provider == "local-bge" and Reranker._instance:
            # FlagReranker 的输入是 (query, passage) 元组列表
            pairs = [[query, p] for p in passages]
            scores = Reranker._instance.compute_score(pairs)
            # 处理可能的单标量返回
            return scores if isinstance(scores, list) else [float(scores)]

        if self.provider == "mock" or not self.endpoint:
            return self._generic_rerank_logic(query, passages)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"query": query, "passages": passages}
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # 假设你的模型服务返回格式为 {"scores": [0.9, 0.1, ...]}
                return data.get("scores") or []
        except Exception as error:
            print(f"WARNING - HTTP rerank failed; using lexical fallback: {redact_secrets(error)}")
            return self._generic_rerank_logic(query, passages) # 异常降级

    def _extract_vertex_scores(self, predictions: Any) -> List[float]:
        """从 Vertex 自定义容器的常见返回格式中提取 rerank 分数。"""
        if isinstance(predictions, dict):
            for key in ("scores", "logits", "predictions"):
                if key in predictions:
                    return self._extract_vertex_scores(predictions[key])
            if "score" in predictions:
                return [float(predictions["score"])]
            return []

        if not isinstance(predictions, list):
            return [float(predictions)]

        if not predictions:
            return []

        if all(isinstance(item, (int, float)) for item in predictions):
            return [float(item) for item in predictions]

        if len(predictions) == 1 and isinstance(predictions[0], (list, dict)):
            return self._extract_vertex_scores(predictions[0])

        scores = []
        for pred in predictions:
            if isinstance(pred, dict):
                if "score" in pred:
                    scores.append(float(pred["score"]))
                elif "scores" in pred:
                    extracted = self._extract_vertex_scores(pred["scores"])
                    scores.append(float(extracted[0]) if extracted else 0.0)
                elif "logits" in pred:
                    extracted = self._extract_vertex_scores(pred["logits"])
                    scores.append(float(extracted[0]) if extracted else 0.0)
                else:
                    scores.append(0.0)
            elif isinstance(pred, list):
                extracted = self._extract_vertex_scores(pred)
                scores.append(float(extracted[0]) if extracted else 0.0)
            else:
                scores.append(float(pred))
        return scores

    def _generic_rerank_logic(self, query: str, passages: List[str]) -> List[float]:
        """
        通用重排逻辑：使用词频重叠度模拟评分，不依赖特定硬件。
        """
        scores = []
        q_terms = set(_tokenize(query))
        for passage in passages:
            p_terms = _tokenize(passage)
            if not p_terms:
                scores.append(0.0)
                continue
            intersection = q_terms.intersection(set(p_terms))
            score = (len(intersection) / (len(q_terms) or 1))
            scores.append(float(score))
        return scores

    def _multi_route_scores(self, query: str, passages: List[dict]) -> List[float]:
        """多路召回融合：召回得分 + 词面(重叠) + 语义评分的混合分数。"""
        texts = [p["text"] for p in passages]
        lexical_scores = [_overlap_score(query, text) for text in texts]
        recall_scores = [float(p.get("recall_score", 0.0)) for p in passages]

        # 从原来的本地模拟逻辑改为调用我们新增的模型方法
        try:
            semantic_scores = self._call_model_rerank(query, texts)
        except Exception as error:
            print(f"WARNING - Rerank model failed; using lexical fallback: {redact_secrets(error)}")
            semantic_scores = self._generic_rerank_logic(query, texts)

        recall_weight = 0.2 if any(score > 0 for score in recall_scores) else 0.0
        blended_scores = []
        for recall, lexical, semantic in zip(recall_scores, lexical_scores, semantic_scores):
            combined = self.weight * semantic + (1 - self.weight) * lexical
            blended_scores.append((1 - recall_weight) * combined + recall_weight * recall)
        return blended_scores

    def rerank(self, query: str, passages: List[dict]) -> List[dict]:
        """
        多路召回 + 二次精排：
        1) 初始召回已由向量检索得到候选。
        2) 结合词面重叠与模型语义得分进行精排。
        3) 返回 Top-K 结果，目标提升 Top-5 命中率（需基准测试验证）。
        """
        if not passages:
            return []

        scores = self._multi_route_scores(query, passages)

        for idx, passage in enumerate(passages):
            passage["rerank_score"] = scores[idx]

        # 按优化后的得分降序排列
        passages.sort(key=lambda p: p.get("rerank_score", 0.0), reverse=True)
        return passages
