#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS = (
    "Summarize the uploaded knowledge base in one paragraph.",
    "What are the most important policy requirements?",
    "List the key exceptions and approval steps.",
    "Which teams are responsible for this process?",
)
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass
class Sample:
    ok: bool
    status: int
    latency_ms: float
    error: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HTTP load tests against the Go backend or Python pipeline and print QPS."
    )
    parser.add_argument(
        "target",
        choices=("backend", "pipeline"),
        help="backend hits /api/query with JWT auth; pipeline hits /query directly.",
    )
    parser.add_argument(
        "--scenario",
        choices=("query", "health"),
        default="query",
        help="query measures end-to-end RAG; health measures service HTTP baseline.",
    )
    parser.add_argument("--url", default=None, help="Base URL. Defaults: backend=http://localhost:8080, pipeline=http://localhost:9000")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers.")
    parser.add_argument("--requests", type=int, default=20, help="Total measured requests. Ignored when --duration is set.")
    parser.add_argument("--duration", type=float, default=0.0, help="Measured duration in seconds. Overrides --requests.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests before measurement.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds.")
    parser.add_argument("--top-k", type=int, default=10, help="RAG retrieval top_k.")
    parser.add_argument("--user-id", default="load_test", help="Pipeline user_id.")
    parser.add_argument("--username", default="load_test", help="Backend login username.")
    parser.add_argument("--password", default="load_test", help="Backend login password.")
    parser.add_argument("--question", action="append", default=[], help="Question to cycle through. Can be repeated.")
    parser.add_argument("--questions-file", type=Path, default=None, help="JSON list of strings or objects with a question field.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any measured request fails.")
    args = parser.parse_args()

    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.duration <= 0 and args.requests <= 0:
        parser.error("--requests must be positive when --duration is not set")

    questions = load_questions(args)
    base_url = (args.url or default_url(args.target)).rstrip("/")
    token = (
        login_backend(base_url, args.username, args.password, args.timeout)
        if args.target == "backend" and args.scenario == "query"
        else ""
    )

    if args.warmup > 0:
        print(f"Warmup: {args.warmup} requests", flush=True)
        warmup_samples = run_fixed_requests(args, base_url, token, questions, args.warmup)
        failed = sum(1 for item in warmup_samples if not item.ok)
        if failed:
            print(f"Warmup failures: {failed}/{len(warmup_samples)}", flush=True)

    print(
        f"Load test: target={args.target} url={base_url} concurrency={args.concurrency} "
        f"scenario={args.scenario} mode={'duration' if args.duration > 0 else 'requests'}",
        flush=True,
    )
    started = time.perf_counter()
    if args.duration > 0:
        samples = run_for_duration(args, base_url, token, questions)
    else:
        samples = run_fixed_requests(args, base_url, token, questions, args.requests)
    elapsed = time.perf_counter() - started

    report = build_report(args, base_url, samples, elapsed)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.output}")

    return 2 if args.fail_on_error and report["errors"] else 0


def load_questions(args: argparse.Namespace) -> list[str]:
    questions = list(args.question)
    if args.questions_file:
        data = json.loads(args.questions_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--questions-file must contain a JSON list")
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and isinstance(item.get("question"), str):
                questions.append(item["question"])
            else:
                raise ValueError("Each question item must be a string or an object with a question field")
    return [item.strip() for item in questions if item.strip()] or list(DEFAULT_QUESTIONS)


def default_url(target: str) -> str:
    return "http://localhost:8080" if target == "backend" else "http://localhost:9000"


def login_backend(base_url: str, username: str, password: str, timeout: float) -> str:
    response = post_json(
        f"{base_url}/api/login",
        {"username": username, "password": password},
        timeout=timeout,
    )
    token = str(response.get("token") or "")
    if not token:
        raise RuntimeError("Backend login did not return a token")
    return token


def run_fixed_requests(
    args: argparse.Namespace,
    base_url: str,
    token: str,
    questions: list[str],
    total_requests: int,
) -> list[Sample]:
    counter = AtomicCounter(total_requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(worker_fixed, args, base_url, token, questions, counter)
            for _ in range(args.concurrency)
        ]
        return flatten(future.result() for future in concurrent.futures.as_completed(futures))


def worker_fixed(
    args: argparse.Namespace,
    base_url: str,
    token: str,
    questions: list[str],
    counter: "AtomicCounter",
) -> list[Sample]:
    samples: list[Sample] = []
    while True:
        index = counter.next_index()
        if index is None:
            return samples
        samples.append(send_query(args, base_url, token, questions[index % len(questions)], index))


def run_for_duration(
    args: argparse.Namespace,
    base_url: str,
    token: str,
    questions: list[str],
) -> list[Sample]:
    stop_at = time.perf_counter() + args.duration
    indexer = AtomicSequence()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(worker_duration, args, base_url, token, questions, stop_at, indexer)
            for _ in range(args.concurrency)
        ]
        return flatten(future.result() for future in concurrent.futures.as_completed(futures))


def worker_duration(
    args: argparse.Namespace,
    base_url: str,
    token: str,
    questions: list[str],
    stop_at: float,
    indexer: "AtomicSequence",
) -> list[Sample]:
    samples: list[Sample] = []
    while time.perf_counter() < stop_at:
        index = indexer.next()
        samples.append(send_query(args, base_url, token, questions[index % len(questions)], index))
    return samples


def send_query(args: argparse.Namespace, base_url: str, token: str, question: str, index: int) -> Sample:
    if args.scenario == "health":
        started = time.perf_counter()
        try:
            get_json(f"{base_url}/health", timeout=args.timeout)
            return Sample(ok=True, status=200, latency_ms=elapsed_ms(started))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            return Sample(ok=False, status=exc.code, latency_ms=elapsed_ms(started), error=body)
        except Exception as exc:
            return Sample(ok=False, status=0, latency_ms=elapsed_ms(started), error=str(exc))

    if args.target == "backend":
        url = f"{base_url}/api/query"
        payload = {
            "question": question,
            "top_k": args.top_k,
            "session_id": f"load-{int(time.time())}-{index}",
        }
        headers = {"Authorization": f"Bearer {token}"}
    else:
        url = f"{base_url}/query"
        payload = {
            "user_id": args.user_id,
            "top_k": args.top_k,
            "messages": [{"role": "user", "content": question}],
        }
        headers = {}

    started = time.perf_counter()
    try:
        post_json(url, payload, timeout=args.timeout, headers=headers)
        return Sample(ok=True, status=200, latency_ms=elapsed_ms(started))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        return Sample(ok=False, status=exc.code, latency_ms=elapsed_ms(started), error=body)
    except Exception as exc:
        return Sample(ok=False, status=0, latency_ms=elapsed_ms(started), error=str(exc))


def post_json(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with LOCAL_OPENER.open(request, timeout=timeout) as response:
        content = response.read().decode("utf-8")
    return json.loads(content) if content else {}


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method="GET")
    with LOCAL_OPENER.open(request, timeout=timeout) as response:
        content = response.read().decode("utf-8")
    return json.loads(content) if content else {}


def build_report(args: argparse.Namespace, base_url: str, samples: list[Sample], elapsed: float) -> dict[str, Any]:
    latencies = sorted(item.latency_ms for item in samples)
    successes = sum(1 for item in samples if item.ok)
    errors = len(samples) - successes
    status_counts: dict[str, int] = {}
    for item in samples:
        status_counts[str(item.status)] = status_counts.get(str(item.status), 0) + 1

    return {
        "target": args.target,
        "scenario": args.scenario,
        "url": base_url,
        "concurrency": args.concurrency,
        "requests": len(samples),
        "successes": successes,
        "errors": errors,
        "error_rate": errors / len(samples) if samples else 0.0,
        "elapsed_seconds": elapsed,
        "qps": successes / elapsed if elapsed > 0 else 0.0,
        "throughput_rps": len(samples) / elapsed if elapsed > 0 else 0.0,
        "latency_ms": latency_summary(latencies),
        "status_counts": status_counts,
        "sample_errors": sample_errors(samples),
    }


def latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": latencies[0],
        "avg": statistics.fmean(latencies),
        "p50": percentile(latencies, 50),
        "p90": percentile(latencies, 90),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "max": latencies[-1],
    }


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile_value / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    fraction = rank - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def print_report(report: dict[str, Any]) -> None:
    latency = report["latency_ms"]
    print("\n=== QPS Benchmark ===")
    print(f"target:       {report['target']}")
    print(f"scenario:     {report['scenario']}")
    print(f"requests:     {report['requests']} success={report['successes']} errors={report['errors']}")
    print(f"elapsed:      {report['elapsed_seconds']:.2f}s")
    print(f"qps:          {report['qps']:.2f} successful req/s")
    print(f"throughput:   {report['throughput_rps']:.2f} total req/s")
    print(f"error_rate:   {report['error_rate'] * 100:.2f}%")
    print(
        "latency_ms:   "
        f"avg={latency['avg']:.2f} p50={latency['p50']:.2f} p90={latency['p90']:.2f} "
        f"p95={latency['p95']:.2f} p99={latency['p99']:.2f} max={latency['max']:.2f}"
    )
    print(f"statuses:     {report['status_counts']}")
    if report["sample_errors"]:
        print("sample_errors:")
        for item in report["sample_errors"]:
            print(f"  status={item['status']} error={item['error']}")


def sample_errors(samples: list[Sample]) -> list[dict[str, Any]]:
    errors = []
    seen = set()
    for item in samples:
        if item.ok:
            continue
        key = (item.status, item.error)
        if key in seen:
            continue
        seen.add(key)
        errors.append(asdict(item))
        if len(errors) >= 5:
            break
    return errors


def flatten(chunks: Any) -> list[Sample]:
    samples: list[Sample] = []
    for chunk in chunks:
        samples.extend(chunk)
    return samples


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class AtomicCounter:
    def __init__(self, total: int) -> None:
        self.total = total
        self.value = 0
        self.lock = threading.Lock()

    def next_index(self) -> int | None:
        with self.lock:
            if self.value >= self.total:
                return None
            index = self.value
            self.value += 1
            return index


class AtomicSequence:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()

    def next(self) -> int:
        with self.lock:
            index = self.value
            self.value += 1
            return index


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
