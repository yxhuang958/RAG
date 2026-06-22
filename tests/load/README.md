# QPS load tests

`benchmark_qps.py` benchmarks the Go backend or Python pipeline and prints successful QPS, total throughput, error rate, status counts, and latency percentiles.

Start services first:

```bash
./scripts/dev.sh
```

Run pipeline load test:

```bash
python tests/load/benchmark_qps.py pipeline --concurrency 4 --requests 20
```

Run backend load test:

```bash
python tests/load/benchmark_qps.py backend --concurrency 4 --requests 20
```

Measure HTTP baseline without invoking RAG or LLM:

```bash
python tests/load/benchmark_qps.py backend --scenario health --concurrency 32 --duration 30
python tests/load/benchmark_qps.py pipeline --scenario health --concurrency 32 --duration 30
```

Duration mode:

```bash
python tests/load/benchmark_qps.py pipeline --concurrency 8 --duration 60 --output tests/load/pipeline_qps.json
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
- `--scenario health` calls `/health` and is useful as a baseline; `--scenario query` is the default end-to-end RAG path.
- For RAG + LLM endpoints, QPS is usually bounded by the LLM provider, embedding/rerank model latency, and Milvus. Treat local numbers as environment-specific, not an absolute capacity claim.
