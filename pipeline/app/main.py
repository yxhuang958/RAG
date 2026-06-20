from __future__ import annotations

import hashlib
import io
import re
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from .config import get_settings
from .document_store import DocumentStore
from .embedding import EmbeddingClient
from .milvus_store import insert_chunks
from .runtime import get_collection, get_pipeline, warmup
from .text_utils import chunk_text

app = FastAPI(title="RAG Pipeline")
settings = get_settings()
document_store = DocumentStore(settings.document_store_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = settings.retrieve_top_k
    user_id: str


class WarmupRequest(BaseModel):
    user_id: str


@app.get("/")
def root():
    return {"message": "RAG Pipeline Service is running"}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/warmup")
def warmup_endpoint(payload: WarmupRequest) -> dict:
    user_id = _normalize_user_id(payload.user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return warmup(user_id)


@app.post("/ingest")
def ingest(file: UploadFile = File(...), user_id: str = Form(...)) -> dict:
    """处理上传的 PDF,进行解析、切片、向量化并存入 Milvus"""
    user_id = _normalize_user_id(user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # 1. 校验文件格式
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2. 读取文件内容到内存
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # 3. 使用 PdfReader 提取文本
    reader = PdfReader(io.BytesIO(content))
    text_parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            text_parts.append(text)

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="No extractable text")

    # 4. 文本切片 (Chunking)
    chunks = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to create chunks")

    file_hash = _sha256_hex(content)
    content_hash = _sha256_text("\n\n".join(chunks))
    doc_id = _build_doc_id(user_id, file.filename, content_hash)
    document, is_new = document_store.reserve(
        user_id=user_id,
        content_hash=content_hash,
        file_hash=file_hash,
        doc_id=doc_id,
        filename=file.filename,
        chunks=len(chunks),
    )
    if not is_new:
        if document.get("status") == "ready" and not document.get("chunk_texts"):
            document = document_store.attach_chunk_texts(user_id, content_hash, chunks) or document
        return {
            "status": "duplicate" if document.get("status") == "ready" else document.get("status"),
            "doc_id": document.get("doc_id"),
            "user_id": document.get("user_id"),
            "filename": document.get("filename"),
            "chunks": document.get("chunks", 0),
            "collection": document.get("collection", settings.milvus_collection),
            "content_hash": content_hash,
            "duplicate": document.get("status") == "ready",
        }

    # 5. 调用 Embedding 接口进行向量化
    try:
        embedding = EmbeddingClient()
        vectors = embedding.embed(chunks)

        # 6. 获取已缓存的 Milvus collection
        collection = get_collection(len(vectors[0]))

        # 7. 写入 Milvus
        ids = insert_chunks(collection, user_id, doc_id, len(chunks), vectors)
        document = document_store.mark_ready(
            content_hash,
            user_id=user_id,
            collection=settings.milvus_collection,
            ids=ids,
            chunk_texts=chunks,
        )
    except Exception as exc:
        document_store.mark_failed(user_id, content_hash, str(exc))
        raise

    return {
        "status": "ok",
        "doc_id": document["doc_id"],
        "user_id": user_id,
        "chunks": len(chunks),
        "collection": settings.milvus_collection,
        "content_hash": content_hash,
        "ids": ids,
    }


@app.post("/query")
def query(payload: QueryRequest) -> dict:
    """接收完整会话消息，最后一条消息作为本轮新问题。"""
    total_start = time.perf_counter()
    start = time.perf_counter()

    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages are required")
    user_id = _normalize_user_id(payload.user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    messages = [
        {"role": item.role.strip().lower(), "content": item.content.strip()}
        for item in payload.messages
        if item.content.strip()
    ]
    if not messages:
        raise HTTPException(status_code=400, detail="Messages are empty")

    question = messages[-1]["content"]
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be a user question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    _log_query_timing("request_parse", start, f"messages={len(messages)} top_k={payload.top_k}")

    # 1. 准备 Milvus 集合
    start = time.perf_counter()
    collection = get_collection(settings.embedding_dim)
    _log_query_timing("milvus_prepare", start, f"collection={settings.milvus_collection}")

    # 2. 运行无状态 QA Pipeline (检索 -> 重排 -> 构造 Prompt -> 调用 LLM)
    start = time.perf_counter()
    pipeline = get_pipeline(user_id)
    result = pipeline.answer(collection, messages, payload.top_k)
    _log_query_timing("pipeline_answer", start)
    _log_query_timing("request_total", total_start)

    return {
        "answer": result["answer"],      # LLM 生成的最终答案
        "contexts": result["contexts"],  # 用于生成的参考原文片段（已重排）
    }


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_doc_id(user_id: str, filename: str, content_hash: str) -> str:
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._-")
    safe_stem = safe_stem or "document"
    safe_user = _normalize_user_id(user_id) or "user"
    return f"{safe_user}-{safe_stem}-{content_hash[:12]}"


def _normalize_user_id(user_id: str | None) -> str:
    if not user_id:
        return ""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", user_id.strip()).strip("._-")


def _log_query_timing(stage: str, start: float, extra: str = "") -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000
    suffix = f" {extra}" if extra else ""
    print(f"TIMING query {stage}={elapsed_ms:.2f}ms{suffix}", flush=True)
