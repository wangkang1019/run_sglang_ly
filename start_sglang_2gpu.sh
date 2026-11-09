#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/.sglang-run"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$RUN_DIR/supervisor.pid"
PYTHON_BIN="/usr/local/bin/python"
SGLANG_BIN="/usr/local/bin/sglang"

usage() {
  cat <<'EOF'
Usage:
  ./start_sglang_2gpu.sh /absolute/path/to/model

Optional environment variables:
  SGLANG_HTTP_PORT=30001        HTTP port
  SERVED_MODEL_NAME=model-name   API model name (defaults to model directory name)
  MODELSCOPE_MODEL_ID=Qwen/Qwen3.8-27B
                                ModelScope repo to download when the local model is absent/incomplete
  MODELSCOPE_CLI=/root/miniconda3/envs/sglang/bin/ms-hub
                                ModelScope Hub CLI
  MODELSCOPE_MAX_WORKERS=4      parallel ModelScope file downloads
  TARGET_GPU_UTIL=60            target average utilization across both cards
  UTIL_HYSTERESIS=10            adjust only outside target +/- this value
  INITIAL_CONCURRENCY=2          starting concurrent requests
  MAX_CONCURRENCY=16             upper limit for concurrent requests
  MAX_TOKENS=256                 generated tokens per request
  SGLANG_CONTEXT_LENGTH=8192     server context limit
  SAMPLE_INTERVAL=10             utilization sampling interval in seconds
  STARTUP_TIMEOUT=1800           max wait for model load and warmup
  PROMPT_FILE=/path/prompts.txt  optional one-prompt-per-line workload file

By default the runner generates local synthetic prompts to keep the model
serving continuously. Set PROMPT_FILE to use your own prompts instead.
EOF
}

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  usage
  [[ $# -eq 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]] && exit 0
  exit 2
fi

MODEL_PATH="$1"
for required in "$PYTHON_BIN" "$SGLANG_BIN"; do
  if [[ ! -x "$required" ]]; then
    echo "Required executable not found: $required" >&2
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi/PPU-SMI compatible command not found in PATH" >&2
  exit 1
fi

MODEL_PATH="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$MODEL_PATH")"
if [[ "$MODEL_PATH" == "/" ]]; then
  echo "Refusing to use / as the model directory." >&2
  exit 2
fi
if [[ -e "$MODEL_PATH" && ! -d "$MODEL_PATH" ]]; then
  echo "Model path exists but is not a directory: $MODEL_PATH" >&2
  exit 2
fi
MODEL_PARENT="$(dirname -- "$MODEL_PATH")"
if ! mkdir -p -- "$MODEL_PARENT"; then
  echo "Cannot create model parent directory: $MODEL_PARENT" >&2
  exit 1
fi
if [[ ! -w "$MODEL_PARENT" ]]; then
  echo "Model parent directory is not writable: $MODEL_PARENT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    if [[ -r "/proc/$OLD_PID/cmdline" ]] && tr '\0' ' ' < "/proc/$OLD_PID/cmdline" | grep -Fq "$SCRIPT_DIR/sglang_keepalive.py"; then
      echo "SGLang runner already active (PID $OLD_PID)."
      echo "Logs: $LOG_DIR/runner.log and $LOG_DIR/server.log"
      exit 0
    fi
  fi
  rm -f "$PID_FILE"
fi

export SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30001}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename -- "$MODEL_PATH")}"
export MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-Qwen/Qwen3.8-27B}"
export MODELSCOPE_CLI="${MODELSCOPE_CLI:-/root/miniconda3/envs/sglang/bin/ms-hub}"
export MODELSCOPE_MAX_WORKERS="${MODELSCOPE_MAX_WORKERS:-4}"
export TARGET_GPU_UTIL="${TARGET_GPU_UTIL:-60}"
export UTIL_HYSTERESIS="${UTIL_HYSTERESIS:-10}"
export INITIAL_CONCURRENCY="${INITIAL_CONCURRENCY:-2}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8192}"
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export PROMPT_FILE="${PROMPT_FILE:-}"
export SGLANG_BIN
# The runner calls its own local API. Bypass the container's network proxy for loopback.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

nohup setsid "$PYTHON_BIN" "$SCRIPT_DIR/sglang_keepalive.py" \
  --model-path "$MODEL_PATH" \
  >> "$LOG_DIR/runner.log" 2>&1 < /dev/null &
SUPERVISOR_PID=$!
echo "$SUPERVISOR_PID" > "$PID_FILE"

sleep 1
if ! kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Runner exited during startup. See $LOG_DIR/runner.log" >&2
  exit 1
fi

echo "Started detached SGLang TP=2 runner (PID $SUPERVISOR_PID)."
echo "Model: $MODEL_PATH"
echo "ModelScope repo (downloaded if model files are missing): $MODELSCOPE_MODEL_ID"
echo "Target utilization: ${TARGET_GPU_UTIL}% average"
echo "Logs: $LOG_DIR/runner.log and $LOG_DIR/server.log"
echo "Stop with: $SCRIPT_DIR/stop_sglang_2gpu.sh"
