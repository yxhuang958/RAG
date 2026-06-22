# RAG Pipeline

A standalone FastAPI pipeline for embedding, Milvus storage, retrieval, rerank, prompt building, and LLM Q&A. It lives inside the main `RAG` repo as `pipeline/`, but it is still a separate service from the Go API.

## Features
- PDF ingestion (`/ingest`) → chunk → embedding → Milvus storage
- Stateless retrieval + rerank + prompt construction (`/query`)
- Long query preprocessing: separates user-provided temporary context from the actual question before retrieval
- Embedding/LLM/rerank can be wired to CANN SDK or HTTP services
- Rerank is configurable via env (`RERANK_PROVIDER`, `RERANK_TOP_K`, `RERANK_WEIGHT`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## Endpoints
- `POST /ingest` (multipart file field: `file`)
- `POST /query` JSON: `{ "messages": [{ "role": "user", "content": "..." }], "top_k": 6 }`
  - `messages` must include the new question as the last item.
  - If the last message contains pasted background plus a question, the pipeline keeps the background as temporary prompt context and retrieves with a shorter question-focused query.
  - The pipeline does not manage users, sessions, JWTs, or chat storage.

## Configuration (env)

### Milvus
- `MILVUS_URI`, `MILVUS_TOKEN`
- or `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_USER`, `MILVUS_PASSWORD`
- `MILVUS_COLLECTION`

### Embedding
- `EMBEDDING_PROVIDER`: `http` or `mock`
- `EMBEDDING_ENDPOINT`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`
- `EMBEDDING_DIM`

### Rerank
- `RERANK_PROVIDER`: `http` or `overlap`
- `RERANK_ENDPOINT`, `RERANK_API_KEY`
- `RERANK_TOP_K`, `RERANK_WEIGHT`

### LLM
- `LLM_PROVIDER`: `http` or `mock`
- `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`

### Chunking
- `CHUNK_SIZE`, `CHUNK_OVERLAP`

### Query preprocessing
- `QUERY_PREPROCESS_ENABLED` default `true`
- `QUERY_LONG_THRESHOLD_CHARS` default `1200`
- `QUERY_CONTEXT_RETRIEVAL_CHARS` default `400`
- `QUERY_TEMP_CONTEXT_PROMPT_CHARS` default `6000`

## Quick self-check

```bash
python scripts/self_check.py
```
