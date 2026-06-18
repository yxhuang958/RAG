# RAG
RAG System Architecture (React + Go + Python)

## Local Run

Start the three services in separate terminals:

```bash
cd client
npm run dev
```

```bash
cd server
go run main.go
```

```bash
cd pipeline
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

The Go API forwards upload/query requests to the Python pipeline through:

```env
PIPELINE_URL=http://localhost:9000
```

Keep app-layer env values in `RAG/.env`, and keep pipeline internals such as Milvus, embedding, rerank, and LLM settings in `RAG/pipeline/.env`.

## Component
| Component              | Responsibility           |
| ---------------------- | ------------------------ |
| MongoDB                | Business data & metadata |
| Milvus                 | Vector similarity search |
| Redis                  | Task queue & caching     |
| Object Storage (S3/R2) | File storage             |
| Go Gin                 | API + orchestration      |
| Python                 | RAG pipeline             |

## Deployment
| Component     | Deployment                   |
| ------------- | ---------------------------- |
| Frontend      | Vercel / Cloudflare Pages    |
| Go API        | AWS ECS / Cloud Run / Docker |
| Python Worker | AWS ECS / Cloud Run / Docker |
| MongoDB       | MongoDB Atlas                |
| Redis         | Upstash / ElastiCache        |
| Milvus        | Zilliz Cloud                 |
| File Storage  | S3 / Cloudflare R2           |
| LLM           | Cloud API                    |

## Layer

### React Frontend
- UI (upload + Q&A)

---

### Go Gin API Server
- Auth / User
- File metadata
- Conversation history
- Job status
- Quota / rate limit
- Dispatch Python RAG Worker
- `/api/upload` 直接转发 PDF 到 Python pipeline（pipeline 负责写入 Milvus）
- `/api/query` 组装当前用户会话历史后转发到 Python pipeline

#### Pipeline 配置
Go API 会把上传和问答请求转发到独立的 Python RAG pipeline：

- `PIPELINE_URL`：Python pipeline 地址（默认 `http://localhost:9000`）
- `PIPELINE_TIMEOUT`：转发超时时间（默认 `60` 秒）
- `PIPELINE_API_KEY`：可选鉴权 key
- `PIPELINE_API_KEY_HEADER`：鉴权 header（默认 `Authorization`）
- `PIPELINE_API_KEY_PREFIX`：鉴权前缀（默认 `Bearer `）

---

### Python RAG Pipeline
embedding + Milvus + 检索 + prompt + LLM 的实现位于项目内：

- `pipeline/`

Pipeline 保持无状态：Go API 传入完整 `messages`，最后一条是本轮用户问题；pipeline 返回 LLM answer 和 RAG contexts。

---

### Redis Queue

---

### Python RAG Worker
- LangChain pipeline
- Document parsing & chunking
- Embedding & rerank
- LLM call

---

### Storage Layer
- Milvus (Vector DB)
- MongoDB (metadata)
- Object Storage (files)
