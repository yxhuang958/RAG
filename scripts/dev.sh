#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.logs"

mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()

cleanup() {
    local code=$?

    if ((${#PIDS[@]} > 0)); then
        printf '\nStopping services...\n'
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait "${PIDS[@]}" 2>/dev/null || true
    fi

    exit "$code"
}

trap cleanup INT TERM EXIT

start_service() {
    local name="$1"
    local dir="$2"
    local log_file="$3"
    shift 3

    printf 'Starting %-8s -> %s\n' "$name" "$log_file"
    (
        cd "$dir"
        "$@" 2>&1 | sed -u "s/^/[$name] /"
    ) | tee "$log_file" &

    PIDS+=("$!")
    NAMES+=("$name")
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$command_name" >&2
        exit 1
    fi
}

require_command npm
require_command go

if [[ -x "$ROOT_DIR/pipeline/.venv/bin/uvicorn" ]]; then
    UVICORN_CMD=("$ROOT_DIR/pipeline/.venv/bin/uvicorn")
else
    require_command uvicorn
    UVICORN_CMD=(uvicorn)
fi

start_service "pipeline" "$ROOT_DIR/pipeline" "$LOG_DIR/pipeline.log" \
    "${UVICORN_CMD[@]}" app.main:app --host 0.0.0.0 --port "${PIPELINE_PORT:-9000}"

start_service "server" "$ROOT_DIR/server" "$LOG_DIR/server.log" \
    go run main.go

start_service "client" "$ROOT_DIR/client" "$LOG_DIR/client.log" \
    npm run dev -- --host 0.0.0.0

printf '\nAll services started.\n'
printf 'Frontend: http://localhost:5173\n'
printf 'Backend:  http://localhost:%s\n' "${SERVER_PORT:-8080}"
printf 'Pipeline: http://localhost:%s\n' "${PIPELINE_PORT:-9000}"
printf 'Logs:     %s\n' "$LOG_DIR"
printf 'Press Ctrl+C to stop everything.\n\n'

while ((${#PIDS[@]} > 0)); do
    for index in "${!PIDS[@]}"; do
        pid="${PIDS[$index]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            printf '\n%s exited. Stopping the rest.\n' "${NAMES[$index]}"
            exit 1
        fi
    done
    sleep 2
done
