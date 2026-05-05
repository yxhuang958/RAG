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
