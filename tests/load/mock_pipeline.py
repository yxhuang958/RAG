#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockPipelineHandler(BaseHTTPRequestHandler):
    server_version = "MockPipeline/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/query":
            _ = self.read_json_body()
            delay_ms = self.server.delay_ms  # type: ignore[attr-defined]
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            self.write_json(
                HTTPStatus.OK,
                {
                    "answer": self.server.answer,  # type: ignore[attr-defined]
                    "sources": [],
                    "mock": True,
                },
            )
            return

        if self.path in {"/warmup", "/ingest_text"}:
            _ = self.read_json_body()
            self.write_json(HTTPStatus.OK, {"status": "ok", "mock": True})
            return

        self.write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(format, *args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast local mock of the Python pipeline for backend QPS tests."
    )
    parser.add_argument("--check-url", default="", help="Check an already-running mock pipeline URL and exit.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=9100, help="Bind port.")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Optional fixed /query delay.")
    parser.add_argument("--answer", default="mock answer", help="Answer returned by /query.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-request access logs.")
    args = parser.parse_args()

    if args.check_url:
        return check_url(args.check_url.rstrip("/"))

    httpd = ThreadingHTTPServer((args.host, args.port), MockPipelineHandler)
    httpd.delay_ms = args.delay_ms
    httpd.answer = args.answer
    httpd.quiet = args.quiet

    print(
        f"Mock pipeline listening on http://{args.host}:{args.port} "
        f"delay_ms={args.delay_ms}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        httpd.server_close()
    return 0


def check_url(base_url: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    checks = [
        ("GET", "/health", None),
        ("POST", "/warmup", {"user_id": "mock_check"}),
        (
            "POST",
            "/query",
            {
                "user_id": "mock_check",
                "top_k": 10,
                "messages": [{"role": "user", "content": "ping"}],
            },
        ),
    ]
    for method, path, payload in checks:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with opener.open(request, timeout=5) as response:
            content = response.read().decode("utf-8")
        print(f"{method} {base_url + path} -> {response.status} {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
