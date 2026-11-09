#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.sglang-run/supervisor.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No SGLang runner PID file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ ! "$PID" =~ ^[0-9]+$ ]] || ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Runner is not active; removed stale PID file."
  exit 0
fi

if [[ ! -r "/proc/$PID/cmdline" ]] || ! tr '\0' ' ' < "/proc/$PID/cmdline" | grep -Fq "$SCRIPT_DIR/sglang_keepalive.py"; then
  echo "PID $PID is not this SGLang runner; refusing to signal it." >&2
  exit 1
fi

kill -TERM "$PID"
for _ in $(seq 1 150); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "SGLang runner stopped."
    exit 0
  fi
  sleep 1
done

echo "Runner is still stopping. Check $SCRIPT_DIR/logs/runner.log" >&2
exit 1
