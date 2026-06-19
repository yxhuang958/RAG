from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]

def _get_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "rag-pipeline")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "9000"))

    milvus_uri: str | None = os.getenv("MILVUS_URI")
    milvus_token: str | None = os.getenv("MILVUS_TOKEN")
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_user: str | None = os.getenv("MILVUS_USER")
    milvus_password: str | None = os.getenv("MILVUS_PASSWORD")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_documents")

    # MongoDB Configuration
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "rag_db")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local-st") # 修改默认为 local-st
    embedding_endpoint: str | None = os.getenv("EMBEDDING_ENDPOINT")
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    rerank_provider: str = os.getenv("RERANK_PROVIDER", "overlap")
    rerank_endpoint: str | None = os.getenv("RERANK_ENDPOINT")
    rerank_api_key: str | None = os.getenv("RERANK_API_KEY")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3") # 新增
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    rerank_weight: float = float(os.getenv("RERANK_WEIGHT", "0.6"))

    llm_provider: str = os.getenv("LLM_PROVIDER", "http")
    llm_endpoint: str | None = os.getenv("LLM_ENDPOINT")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "llm-model")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    # GCP Vertex AI Settings
    gcp_project_id: str | None = os.getenv("GCP_PROJECT_ID")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    vertex_embedding_endpoint_id: str | None = os.getenv("VERTEX_EMBEDDING_ENDPOINT_ID")
    vertex_rerank_endpoint_id: str | None = os.getenv("VERTEX_RERANK_ENDPOINT_ID")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    document_store_path: str = os.getenv(
        "DOCUMENT_STORE_PATH",
        str(_REPO_ROOT / "server" / "data" / "pipeline_documents.json"),
    )
    debug: bool = _get_bool(os.getenv("DEBUG"), False)


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
