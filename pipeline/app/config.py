from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PIPELINE_ROOT / ".env")
load_dotenv(_REPO_ROOT / ".env")
_VECTOR_METRICS = {"L2", "COSINE", "IP"}

def _get_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _get_vector_metric() -> str:
    metric = (os.getenv("VECTOR_METRIC") or "L2").strip().upper()
    if metric not in _VECTOR_METRICS:
        raise ValueError(f"VECTOR_METRIC must be one of {sorted(_VECTOR_METRICS)}, got {metric!r}")
    return metric


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "rag-pipeline")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 9000)

    milvus_uri: str | None = os.getenv("MILVUS_URI")
    milvus_token: str | None = os.getenv("MILVUS_TOKEN")
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_user: str | None = os.getenv("MILVUS_USER")
    milvus_password: str | None = os.getenv("MILVUS_PASSWORD")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_documents")

    # MongoDB Configuration
    mongo_uri: str = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGODB_DATABASE") or os.getenv("MONGO_DB", "rag_db")
    mongo_documents_collection: str = os.getenv("MONGODB_DOCUMENTS_COLLECTION", "pipeline_documents")
    mongo_chunks_collection: str = os.getenv("MONGODB_CHUNKS_COLLECTION", "pipeline_chunks")
    mongo_timeout_ms: int = _get_int("MONGODB_TIMEOUT_MS", 5000)
    mongo_server_selection_timeout_ms: int = _get_int("MONGODB_SERVER_SELECTION_TIMEOUT_MS", 5000)
    mongo_max_pool_size: int = _get_int("MONGODB_MAX_POOL_SIZE", 100)

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local-st") # 修改默认为 local-st
    embedding_endpoint: str | None = os.getenv("EMBEDDING_ENDPOINT")
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = _get_int("EMBEDDING_DIM", 1024)
    vector_metric: str = _get_vector_metric()
    normalize_embeddings: bool = _get_bool(os.getenv("NORMALIZE_EMBEDDINGS"), False)

    rag_enabled: bool = _get_bool(os.getenv("RAG_ENABLED"), True)
    multi_retrieval_enabled: bool = _get_bool(os.getenv("MULTI_RETRIEVAL_ENABLED"), False)
    multi_retrieval_dense_factor: int = _get_int("MULTI_RETRIEVAL_DENSE_FACTOR", 3)
    multi_retrieval_lexical_factor: int = _get_int("MULTI_RETRIEVAL_LEXICAL_FACTOR", 3)
    multi_retrieval_rrf_k: int = _get_int("MULTI_RETRIEVAL_RRF_K", 60)
    retrieval_neighbor_window: int = _get_int("RETRIEVAL_NEIGHBOR_WINDOW", 1)
    evidence_window_sentences: int = _get_int("EVIDENCE_WINDOW_SENTENCES", 1)
    evidence_candidate_factor: int = _get_int("EVIDENCE_CANDIDATE_FACTOR", 3)
    context_max_per_doc: int = _get_int("CONTEXT_MAX_PER_DOC", 3)
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "overlap")
    rerank_enabled: bool = _get_bool(os.getenv("RERANK_ENABLED"), True)
    rerank_endpoint: str | None = os.getenv("RERANK_ENDPOINT")
    rerank_api_key: str | None = os.getenv("RERANK_API_KEY")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3") # 新增
    rerank_top_k: int = _get_int("RERANK_TOP_K", 5)
    rerank_weight: float = _get_float("RERANK_WEIGHT", 0.6)
    rerank_circuit_breaker_seconds: int = _get_int("RERANK_CIRCUIT_BREAKER_SECONDS", 60)
    rerank_circuit_breaker_failures: int = _get_int("RERANK_CIRCUIT_BREAKER_FAILURES", 1)
    rerank_vertex_retries: int = _get_int("RERANK_VERTEX_RETRIES", 2)
    second_stage_rerank_enabled: bool = _get_bool(os.getenv("SECOND_STAGE_RERANK_ENABLED"), False)
    second_stage_rerank_model: str = os.getenv(
        "SECOND_STAGE_RERANK_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    second_stage_rerank_weight: float = _get_float("SECOND_STAGE_RERANK_WEIGHT", 0.35)
    second_stage_history_messages: int = _get_int("SECOND_STAGE_HISTORY_MESSAGES", 4)
    query_preprocess_enabled: bool = _get_bool(os.getenv("QUERY_PREPROCESS_ENABLED"), True)
    query_long_threshold_chars: int = _get_int("QUERY_LONG_THRESHOLD_CHARS", 1200)
    query_context_retrieval_chars: int = _get_int("QUERY_CONTEXT_RETRIEVAL_CHARS", 400)
    query_temp_context_prompt_chars: int = _get_int("QUERY_TEMP_CONTEXT_PROMPT_CHARS", 6000)

    llm_provider: str = os.getenv("LLM_PROVIDER", "http")
    llm_endpoint: str | None = os.getenv("LLM_ENDPOINT")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "llm-model")
    llm_timeout_seconds: int = _get_int("LLM_TIMEOUT_SECONDS", 60)
    llm_retries: int = _get_int("LLM_RETRIES", 3)

    # GCP Vertex AI Settings
    gcp_project_id: str | None = os.getenv("GCP_PROJECT_ID")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    vertex_embedding_endpoint_id: str | None = os.getenv("VERTEX_EMBEDDING_ENDPOINT_ID")
    vertex_rerank_endpoint_id: str | None = os.getenv("VERTEX_RERANK_ENDPOINT_ID")

    chunk_size: int = _get_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 80)
    document_store_path: str = os.getenv(
        "DOCUMENT_STORE_PATH",
        str(_REPO_ROOT / "server" / "data" / "pipeline_documents.json"),
    )
    document_store_backend: str = os.getenv("DOCUMENT_STORE_BACKEND") or os.getenv("CHAT_STORE_BACKEND", "json")
    debug: bool = _get_bool(os.getenv("DEBUG"), False)
    
    retrieve_top_k: int = _get_int("RETRIEVE_TOPK", 30)

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

"""
Milvus连接细节:
1. 当你执行 connections.connect(alias="default", ...) 时，你并没有创建一个局部变量，
    而是向 PyMilvus 的全局管理器注册了一个名为 "default" 的连接。
2. 后续你调用 utility.has_collection() 或 Collection("name") 时，如果你不指定连接名，
    PyMilvus 默认就会去查那个叫 "default" 的全局连接句柄。
ps: 总结流程
    Test 脚本调用 connect_milvus():
        通过单例 get_settings() 获取云端地址。
        调用 connections.connect。此时, 全局连接池里已经握住了通往 Zilliz 的“梯子”。
    Test 脚本直接调用 Collection(...):
        它不需要你传入 conn 对象，因为它会自动去全局连接池里找那个默认的“梯子”。
"""
