import os
import statistics
import time

from app.config import get_settings
from app.milvus_store import connect_milvus, get_or_create_collection
from app.rerank import Reranker
from app.retriever import Retriever


def _mock_cases():
    return [
        {
            "question": "CANN 的核心优势是什么？",
            "expected_doc_id": "doc_001",
            "candidates": [
                {"doc_id": "doc_010", "text": "random info about cpu", "recall_score": 0.45},
                {
                    "doc_id": "doc_001",
                    "text": "semantic:match CANN accelerates ai workloads",
                    "recall_score": 0.35,
                },
                {"doc_id": "doc_011", "text": "deployment notes", "recall_score": 0.3},
                {"doc_id": "doc_012", "text": "gpu overview", "recall_score": 0.25},
                {"doc_id": "doc_013", "text": "cloud infra", "recall_score": 0.2},
            ],
        },
        {
            "question": "如何部署 Milvus？",
            "expected_doc_id": "doc_002",
            "candidates": [
                {"doc_id": "doc_020", "text": "k8s baseline", "recall_score": 0.44},
                {"doc_id": "doc_021", "text": "milvus docker compose", "recall_score": 0.42},
                {
                    "doc_id": "doc_002",
                    "text": "semantic:match Milvus standalone deployment",
                    "recall_score": 0.3,
                },
                {"doc_id": "doc_022", "text": "vector db overview", "recall_score": 0.28},
                {"doc_id": "doc_023", "text": "storage layer", "recall_score": 0.24},
            ],
        },
    ]


def _evaluate_with_mock(reranker: Reranker) -> tuple[float, float]:
    hits_count = 0
    latencies = []
    cases = _mock_cases()
    for case in cases:
        start_time = time.perf_counter()
        final_results = reranker.rerank(case["question"], case["candidates"])
        latencies.append(time.perf_counter() - start_time)

        top_ids = [res["doc_id"] for res in final_results[:5]]
        if case["expected_doc_id"] in top_ids:
            hits_count += 1

    accuracy = hits_count / len(cases)
    avg_latency = statistics.mean(latencies) * 1000
    return accuracy, avg_latency


def _evaluate_with_milvus(reranker: Reranker) -> tuple[float, float]:
    settings = get_settings()
    connect_milvus()
    collection = get_or_create_collection(settings.embedding_dim)
    retriever = Retriever()

    test_cases = [
        ("CANN 的核心优势是什么？", "doc_001"),
        ("如何部署 Milvus？", "doc_002"),
    ]

    hits_count = 0
    latencies = []
    for question, expected_id in test_cases:
        start_time = time.perf_counter()
        initial_hits = retriever.retrieve(collection, question, top_k=20)
        final_results = reranker.rerank(question, initial_hits)
        latencies.append(time.perf_counter() - start_time)

        top_ids = [res["doc_id"] for res in final_results[:5]]
        if expected_id in top_ids:
            hits_count += 1

    accuracy = hits_count / len(test_cases)
    avg_latency = statistics.mean(latencies) * 1000
    return accuracy, avg_latency


def evaluate_retrieval() -> None:
    """测试检索准确率与性能（对比 Baseline 和 Custom Rerank）"""
    use_milvus = os.getenv("USE_MILVUS", "false").lower() in {"1", "true", "yes"}
    strategies = ["baseline", "custom_model"]

    for strategy in strategies:
        print(f"\n===== 测试策略: {strategy} =====")
        reranker = Reranker()
        reranker.provider = "overlap" if strategy == "baseline" else "custom_model"

        if use_milvus:
            accuracy, avg_latency = _evaluate_with_milvus(reranker)
        else:
            accuracy, avg_latency = _evaluate_with_mock(reranker)

        print(f"准确率 (Top-5 Hit Rate): {accuracy:.2%}")
        print(f"平均端到端耗时: {avg_latency:.2f} ms")


if __name__ == "__main__":
    evaluate_retrieval()
