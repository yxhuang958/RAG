from __future__ import annotations

import io
import uuid
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from .config import get_settings
from .embedding import EmbeddingClient
from .milvus_store import connect_milvus, get_or_create_collection, insert_chunks
from .qa import QAPipeline
from .text_utils import chunk_text

app = FastAPI(title="RAG Pipeline")
settings = get_settings()

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
    top_k: int = 6


@app.get("/")
def root():
    return {"message": "RAG Pipeline Service is running"}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """处理上传的 PDF,进行解析、切片、向量化并存入 Milvus"""
    # 1. 校验文件格式
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2. 读取文件内容到内存
    content = await file.read()
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

    # 5. 调用 Embedding 接口进行向量化
    embedding = EmbeddingClient()
    vectors = embedding.embed(chunks)

    # 6. 连接 Milvus 并获取/创建集合
    connect_milvus()
    collection = get_or_create_collection(len(vectors[0]))

    # 7. 写入 Milvus
    doc_id = f"{file.filename}-{uuid.uuid4().hex[:8]}"
    ids = insert_chunks(collection, doc_id, chunks, vectors)

    return {
        "status": "ok",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "collection": settings.milvus_collection,
        "ids": ids,
    }


@app.post("/query")
async def query(payload: QueryRequest) -> dict:
    """接收完整会话消息，最后一条消息作为本轮新问题。"""
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages are required")

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

    # 1. 准备 Milvus 集合
    connect_milvus()
    collection = get_or_create_collection(settings.embedding_dim)

    # 2. 运行无状态 QA Pipeline (检索 -> 重排 -> 构造 Prompt -> 调用 LLM)
    pipeline = QAPipeline()
    result = pipeline.answer(collection, messages, payload.top_k)

    return {
        "answer": result["answer"],      # LLM 生成的最终答案
        "contexts": result["contexts"],  # 用于生成的参考原文片段（已重排）
    }
