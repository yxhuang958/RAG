# Long Multi-hop Hallucination Eval

This suite targets long-document, multi-hop RAG hallucination behavior. It is intentionally separate from `tests/rag_tests`, which is mostly short single-hop policy QA.

## Dataset

The preparation script uses Hugging Face `datasets.load_dataset` to load a subset of the public HotpotQA distractor validation set:

- multi-hop questions
- at least two supporting Wikipedia pages
- gold answer
- supporting facts with sentence-level evidence

The script converts selected Wikipedia contexts into long PDF documents so the current pipeline can ingest them through `/ingest`.

## Prepare Data

```bash
python -m pip install -r tests/rag_tests_1/requirements.txt
python tests/rag_tests_1/prepare_hotpotqa.py --limit 80
```

Useful options:

```bash
python tests/rag_tests_1/prepare_hotpotqa.py \
  --dataset-name hotpotqa/hotpot_qa \
  --dataset-config distractor \
  --dataset-split validation \
  --scan-limit 2000 \
  --limit 80
```

Outputs:

- `tests/rag_tests_1/data/*.pdf`
- `tests/rag_tests_1/qa_eval/hotpotqa_long_multihop.json`

## Run Eval

Start the pipeline first, then:

```bash
python tests/rag_tests_1/evaluate_hotpotqa.py \
  --pipeline-url http://localhost:9000 \
  --user-id hotpot_eval \
  --ingest \
  --top-k 10 \
  --output tests/rag_tests_1/hotpotqa_eval_report.json
```

For repeated runs with the same `user_id`, omit `--ingest`.

## Metrics

- `Answer EM/F1`: HotpotQA-style normalized exact match and token F1.
- `SupportDocRecall@K`: fraction of gold supporting documents retrieved.
- `JointDocRecall@K`: whether all supporting documents were retrieved.
- `DocMRR`: reciprocal rank of the first retrieved gold supporting document.
- `SupportSentRecall@K`: fraction of gold supporting evidence sentences present in retrieved contexts.
- `JointSentRecall@K`: whether all supporting evidence sentences were retrieved.
- `AnswerSupported`: whether the generated answer is directly present in retrieved evidence.
- `UnsupportedClaimRate`: whether the answer contains named entities, acronyms, dates, or numbers not present in retrieved evidence.
- `HallucinationFlagRate`: a conservative proxy flag combining unsupported claims and low-answer-F1 unsupported answers.

These are automated proxies. They are useful for regression and model comparison, but serious release gates should still inspect a sample of flagged cases manually.
