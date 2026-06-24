# QPS load tests

`benchmark_qps.py` benchmarks the Go backend or Python pipeline and prints successful QPS, total throughput, error rate, status counts, and latency percentiles.

Start services first:

```bash
./scripts/dev.sh
```

`dev.sh` starts the real pipeline, a local mock pipeline, the Go backend, and the frontend. The backend uses the real pipeline by default; load tests can opt into the mock pipeline per request.

Run pipeline load test:

```bash
python tests/load/benchmark_qps.py pipeline --pipeline-mode real --concurrency 4 --requests 20
```

Run backend load test:

```bash
python tests/load/benchmark_qps.py backend --pipeline-mode real --concurrency 4 --requests 20
```

Measure HTTP baseline without invoking RAG or LLM:

```bash
python tests/load/benchmark_qps.py backend --scenario health --concurrency 32 --duration 30
python tests/load/benchmark_qps.py pipeline --scenario health --concurrency 32 --duration 30
```

Estimate backend query-path upper bound with a fast mock pipeline:

```bash
python tests/load/benchmark_qps.py backend \
  --url http://localhost:8080 \
  --pipeline-mode mock \
  --scenario query \
  --concurrency 32 \
  --duration 30 \
  --username load_test_backend_mock \
  --password load_test_backend_mock \
  --output tests/load/backend_mock_pipeline_qps.json
```

This keeps the backend's authenticated `/api/query` path, JSON processing, chat history writes, and outbound HTTP call intact while replacing the expensive RAG/LLM pipeline with a near-zero-latency local service.

Run direct mock pipeline benchmark:

```bash
python tests/load/benchmark_qps.py pipeline \
  --pipeline-mode mock \
  --concurrency 32 \
  --duration 30 \
  --output tests/load/pipeline_mock_qps.json
```

The JSON chat store is useful for local development, but it is not a production-grade high-QPS state store. The store writes asynchronously by default, so it should not dominate request latency; use a real store such as Redis or Postgres before treating chat persistence numbers as production capacity.

Duration mode:

```bash
python tests/load/benchmark_qps.py pipeline --pipeline-mode real --concurrency 8 --duration 60 --output tests/load/pipeline_real_qps.json
```

Use realistic questions:

```bash
python tests/load/benchmark_qps.py backend \
  --questions-file tests/rag_tests/qa_eval/reference_questions_answers.json \
  --concurrency 4 \
  --requests 40
```

Notes:

- `qps` counts successful requests per second.
- `throughput` counts all requests per second, including failures.
- Backend mode logs in through `/api/login` and then tests `/api/query`.
- Pipeline mode calls `/query` directly.
- `--pipeline-mode real` uses the normal pipeline. `--pipeline-mode mock` uses the local mock pipeline; for backend tests this is selected per request.
- `--scenario health` calls `/health` and is useful as a baseline; `--scenario query` is the default end-to-end RAG path.
- For backend capacity with an unconstrained pipeline, use `backend --pipeline-mode mock`; `backend --scenario health` is too shallow because it skips auth, chat history, JSON payload work, and backend-to-pipeline HTTP.
- For RAG + LLM endpoints, QPS is usually bounded by the LLM provider, embedding/rerank model latency, and Milvus. Treat local numbers as environment-specific, not an absolute capacity claim.
