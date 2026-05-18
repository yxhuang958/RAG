# RAG
RAG System Architecture (Go + Python + MongoDB)

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
- `/api/upload` 直接转发 PDF 到云端 CANN 解析服务（服务负责写入 Milvus）
- `/api/query` 直接转发检索 + 问答请求到云端 CANN 服务

#### Upload + CANN 云端配置
Go API 会将上传的 PDF 以 multipart 方式发送到云端：

- `CANN_INGEST_URL`：云端 CANN 接口地址（必填）
- `CANN_INGEST_TIMEOUT_SECONDS`：超时时间（默认 `120`）
- `CANN_API_KEY`：可选鉴权 key
- `CANN_API_KEY_HEADER`：鉴权 header（默认 `Authorization`）
- `CANN_API_KEY_PREFIX`：鉴权前缀（默认 `Bearer `）

#### Query + CANN 云端配置

- `CANN_QUERY_URL`：云端检索/问答接口地址（必填）
- `CANN_QUERY_TIMEOUT_SECONDS`：超时时间（默认 `60`）
- `CANN_API_KEY` / `CANN_API_KEY_HEADER` / `CANN_API_KEY_PREFIX` 同上

---

### External CANN Pipeline
云端 CANN 的 embedding + Milvus + 检索 + prompt + LLM 的实现独立在项目外：

- `../cann-rag-pipeline`

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
