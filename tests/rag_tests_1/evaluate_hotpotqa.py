#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_QA_PATH = ROOT / "qa_eval" / "hotpotqa_long_multihop.json"
DEFAULT_DATA_DIR = ROOT / "data"
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

ABSTAIN_PATTERNS = (
    "i don't know",
    "do not know",
    "cannot determine",
    "not enough information",
    "not provided",
    "unknown",
    "不知道",
    "无法确定",
)
FACT_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){0,5}|"
    r"[A-Z]{2,}|"
    r"\d{1,4}(?:[.,:/-]\d{1,4})*(?:\s*(?:BC|AD|km|miles|years|year|percent|%))?)\b"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate long-document multi-hop RAG hallucination risk.")
    parser.add_argument("--pipeline-url", default="http://localhost:9000")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--user-id", default=f"hotpot_eval_{int(time.time())}")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=ROOT / "hotpotqa_eval_report.json")
    args = parser.parse_args()

    items = load_json(args.qa)
    if args.limit > 0:
        items = items[: args.limit]

    if args.ingest:
        ingest_all(args.pipeline_url, args.data_dir, args.user_id, args.timeout)

    cases = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['id']} {item['question']}", flush=True)
        try:
            response = query_pipeline(args.pipeline_url, args.user_id, item["question"], args.top_k, args.timeout)
            cases.append(evaluate_case(item, response, args.top_k))
        except Exception as exc:
            cases.append(error_case(item, str(exc)))

    report = build_report(cases, args.top_k, args.user_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Report written to {args.output}")
    return 0 if report["summary"]["errors"] == 0 else 2


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_all(base_url: str, data_dir: Path, user_id: str, timeout: float) -> None:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {data_dir}. Run prepare_hotpotqa.py first.")

    print(f"Ingesting {len(pdfs)} PDFs as user_id={user_id}", flush=True)
    for index, pdf in enumerate(pdfs, start=1):
        result = ingest_pdf(base_url, user_id, pdf, timeout)
        print(f"  [{index}/{len(pdfs)}] {pdf.name}: {result.get('status', 'unknown')}", flush=True)


def ingest_pdf(base_url: str, user_id: str, pdf_path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"----rag-long-eval-{uuid.uuid4().hex}"
    body = bytearray()
    add_form_field(body, boundary, "user_id", user_id)
    add_file_field(body, boundary, "file", pdf_path.name, pdf_path.read_bytes(), "application/pdf")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ingest",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return read_json(request, timeout)


def add_form_field(body: bytearray, boundary: str, name: str, value: str) -> None:
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
    body.extend(value.encode("utf-8"))
    body.extend(b"\r\n")


def add_file_field(body: bytearray, boundary: str, name: str, filename: str, content: bytes, content_type: str) -> None:
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(content)
    body.extend(b"\r\n")


def query_pipeline(base_url: str, user_id: str, question: str, top_k: int, timeout: float) -> dict[str, Any]:
    payload = {"user_id": user_id, "top_k": top_k, "messages": [{"role": "user", "content": question}]}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return read_json(request, timeout)


def read_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with LOCAL_OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def evaluate_case(item: dict[str, Any], response: dict[str, Any], top_k: int) -> dict[str, Any]:
    contexts = (response.get("contexts") or [])[:top_k]
    answer = str(response.get("answer") or "")
    evidence_text = "\n".join(str(context.get("text") or "") for context in contexts)

    expected_docs = [source_to_doc_key(source) for source in item.get("source_files", [])]
    retrieved_docs = [context_to_doc_key(context) for context in contexts]
    doc_ranks = support_doc_ranks(expected_docs, contexts)
    support_doc_recall = len({rank["expected"] for rank in doc_ranks}) / len(expected_docs) if expected_docs else 0.0
    support_doc_mrr = 1.0 / min(rank["rank"] for rank in doc_ranks) if doc_ranks else 0.0

    support_sentences = [fact["sentence"] for fact in item.get("supporting_facts", []) if fact.get("sentence")]
    sentence_hits = [sentence_supported(sentence, evidence_text) for sentence in support_sentences]
    support_sentence_recall = mean(1.0 if hit else 0.0 for hit in sentence_hits)

    reference = str(item.get("reference_answer") or "")
    scored_answer = extract_answer_for_scoring(answer)
    clean_answer = strip_citations(scored_answer)
    answer_em = exact_match(scored_answer, reference)
    answer_f1 = token_f1(scored_answer, reference)
    answer_supported = answer_supported_by_context(clean_answer, evidence_text, reference)
    unsupported_claims = unsupported_facts(clean_answer, evidence_text)
    abstained = any(pattern in answer.lower() for pattern in ABSTAIN_PATTERNS)
    hallucination_flag = bool(unsupported_claims) or (not answer_supported and not abstained and answer_f1 < 0.5)

    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "reference_answer": reference,
        "answer": answer,
        "scored_answer": scored_answer,
        "clean_answer": clean_answer,
        "expected_docs": expected_docs,
        "retrieved_docs": retrieved_docs,
        "support_doc_recall_at_k": support_doc_recall,
        "joint_support_doc_recall_at_k": support_doc_recall == 1.0,
        "support_doc_mrr": support_doc_mrr,
        "support_sentence_recall_at_k": support_sentence_recall,
        "joint_support_sentence_recall_at_k": bool(sentence_hits) and all(sentence_hits),
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "answer_supported_by_context": answer_supported,
        "unsupported_claims": unsupported_claims,
        "hallucination_flag": hallucination_flag,
        "abstained": abstained,
        "error": None,
    }


def error_case(item: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "reference_answer": item.get("reference_answer"),
        "answer": "",
        "expected_docs": [source_to_doc_key(source) for source in item.get("source_files", [])],
        "retrieved_docs": [],
        "support_doc_recall_at_k": 0.0,
        "joint_support_doc_recall_at_k": False,
        "support_doc_mrr": 0.0,
        "support_sentence_recall_at_k": 0.0,
        "joint_support_sentence_recall_at_k": False,
        "answer_em": 0.0,
        "answer_f1": 0.0,
        "answer_supported_by_context": False,
        "unsupported_claims": [],
        "hallucination_flag": False,
        "abstained": False,
        "error": error,
    }


def source_to_doc_key(source: str) -> str:
    return normalize_key(Path(source).with_suffix("").as_posix().replace("/", "__"))


def context_to_doc_key(context: dict[str, Any]) -> str:
    raw = " ".join(str(context.get(key) or "") for key in ("doc_id", "filename", "source", "metadata"))
    return normalize_key(raw)


def normalize_key(value: str) -> str:
    value = re.sub(r"\.pdf|\.md", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.lower()).strip("_")
    return value


def support_doc_ranks(expected_docs: list[str], contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = []
    for expected in expected_docs:
        for rank, context in enumerate(contexts, start=1):
            if expected and expected in context_to_doc_key(context):
                ranks.append({"expected": expected, "rank": rank})
                break
    return ranks


def sentence_supported(sentence: str, evidence_text: str) -> bool:
    sentence_norm = normalize_text(sentence)
    evidence_norm = normalize_text(evidence_text)
    if not sentence_norm:
        return False
    if sentence_norm in evidence_norm:
        return True

    sentence_tokens = sentence_norm.split()
    evidence_tokens = set(evidence_norm.split())
    if not sentence_tokens:
        return False
    overlap = sum(1 for token in sentence_tokens if token in evidence_tokens)
    return overlap / len(sentence_tokens) >= 0.85


def answer_supported_by_context(answer: str, evidence_text: str, reference: str = "") -> bool:
    normalized_answer = normalize_answer(answer)
    if normalized_answer in {"yes", "no", "true", "false"}:
        return True
    normalized_evidence = normalize_answer(evidence_text)
    if bool(normalized_answer) and normalized_answer in normalized_evidence:
        return True

    normalized_reference = normalize_answer(reference)
    return bool(normalized_reference) and normalized_reference in normalized_evidence and token_f1(answer, reference) >= 0.5


def unsupported_facts(answer: str, evidence_text: str) -> list[str]:
    evidence_norm = normalize_text(evidence_text)
    claims = sorted({match.group(0).strip() for match in FACT_RE.finditer(answer)})
    unsupported = []
    for claim in claims:
        claim_norm = normalize_text(claim)
        if claim_norm and claim_norm not in evidence_norm:
            unsupported.append(claim)
    return unsupported


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(strip_citations(prediction)) == normalize_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(strip_citations(prediction)).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer_for_scoring(text: str) -> str:
    match = re.search(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip()

    evidence_match = re.search(r"(?im)^\s*Evidence\s*:", text)
    if evidence_match:
        return text[: evidence_match.start()].strip()

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line or text.strip()


def strip_citations(text: str) -> str:
    text = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\(\s*source\s*:\s*\d+\s*\)", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def build_report(cases: list[dict[str, Any]], top_k: int, user_id: str) -> dict[str, Any]:
    successful = [case for case in cases if not case["error"]]
    summary = {
        "user_id": user_id,
        "top_k": top_k,
        "total": len(cases),
        "successful": len(successful),
        "errors": len(cases) - len(successful),
        "answer_em": mean(case["answer_em"] for case in successful),
        "answer_f1": mean(case["answer_f1"] for case in successful),
        "support_doc_recall_at_k": mean(case["support_doc_recall_at_k"] for case in successful),
        "joint_support_doc_recall_at_k": mean(1.0 if case["joint_support_doc_recall_at_k"] else 0.0 for case in successful),
        "support_doc_mrr": mean(case["support_doc_mrr"] for case in successful),
        "support_sentence_recall_at_k": mean(case["support_sentence_recall_at_k"] for case in successful),
        "joint_support_sentence_recall_at_k": mean(
            1.0 if case["joint_support_sentence_recall_at_k"] else 0.0 for case in successful
        ),
        "answer_supported_rate": mean(1.0 if case["answer_supported_by_context"] else 0.0 for case in successful),
        "unsupported_claim_rate": mean(1.0 if case["unsupported_claims"] else 0.0 for case in successful),
        "hallucination_flag_rate": mean(1.0 if case["hallucination_flag"] else 0.0 for case in successful),
        "abstention_rate": mean(1.0 if case["abstained"] else 0.0 for case in successful),
    }
    return {"summary": summary, "cases": cases}


def mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n=== Long Multi-hop RAG Eval Summary ===")
    print(f"total={summary['total']} successful={summary['successful']} errors={summary['errors']}")
    print(f"Answer EM={summary['answer_em']:.3f} F1={summary['answer_f1']:.3f}")
    print(
        f"SupportDocRecall@{summary['top_k']}={summary['support_doc_recall_at_k']:.3f} "
        f"JointDocRecall@{summary['top_k']}={summary['joint_support_doc_recall_at_k']:.3f} "
        f"DocMRR={summary['support_doc_mrr']:.3f}"
    )
    print(
        f"SupportSentRecall@{summary['top_k']}={summary['support_sentence_recall_at_k']:.3f} "
        f"JointSentRecall@{summary['top_k']}={summary['joint_support_sentence_recall_at_k']:.3f}"
    )
    print(
        f"AnswerSupported={summary['answer_supported_rate']:.3f} "
        f"UnsupportedClaimRate={summary['unsupported_claim_rate']:.3f} "
        f"HallucinationFlagRate={summary['hallucination_flag_rate']:.3f} "
        f"AbstentionRate={summary['abstention_rate']:.3f}"
    )
    errors = [case for case in report["cases"] if case["error"]]
    if errors:
        print("\nError samples:")
        for case in errors[:5]:
            print(f"- {case['id']}: {case['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
