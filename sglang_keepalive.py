#!/usr/local/bin/python
"""Run one SGLang TP=2 server and adapt request concurrency to PPU utilization."""

from __future__ import annotations

import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / ".sglang-run"
LOG_DIR = SCRIPT_DIR / "logs"
PID_FILE = RUN_DIR / "supervisor.pid"
SERVER_LOG = LOG_DIR / "server.log"
DEFAULT_MODELSCOPE_CLI = "/root/miniconda3/envs/sglang/bin/ms-hub"
DEFAULT_MODELSCOPE_MODEL_ID = "Qwen/Qwen3.8-27B"

PROMPT_TEMPLATES = (
    "请分步骤分析一个高并发推理服务的性能瓶颈，并给出可执行的优化建议。",
    "请写一篇关于大型语言模型推理服务可靠性的详细说明，覆盖监控、恢复和容量规划。",
    "请比较张量并行、流水线并行和数据并行的适用场景，并给出实例。",
    "请设计一个可复现的机器学习性能评测方案，说明指标、变量和结果解释方式。",
)


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def model_snapshot_is_complete(model_path: Path) -> bool:
    """Return whether a local model directory has config, tokenizer and all indexed weights."""
    config_path = model_path / "config.json"
    if not _is_nonempty_file(config_path):
        return False
    try:
        if not isinstance(json.loads(config_path.read_text(encoding="utf-8")), dict):
            return False
    except (OSError, json.JSONDecodeError):
        return False

    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
    )
    if not any(_is_nonempty_file(model_path / name) for name in tokenizer_files):
        return False

    index_files = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    for index_name in index_files:
        index_path = model_path / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        weight_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in weight_names):
            return False
        shard_names = set(weight_names)
        for shard_name in shard_names:
            shard_path = Path(shard_name)
            if shard_path.is_absolute() or ".." in shard_path.parts:
                return False
            shard_file = model_path / shard_path
            if not _is_nonempty_file(shard_file):
                return False
        return True

    return any(
        _is_nonempty_file(model_path / name)
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


class SGLangRunner:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.port = env_int("SGLANG_HTTP_PORT", 30001)
        self.target_util = env_float("TARGET_GPU_UTIL", 60.0)
        self.hysteresis = env_float("UTIL_HYSTERESIS", 10.0)
        self.initial_concurrency = env_int("INITIAL_CONCURRENCY", 2)
        self.max_concurrency = env_int("MAX_CONCURRENCY", 16)
        self.max_tokens = env_int("MAX_TOKENS", 256)
        self.context_length = env_int("SGLANG_CONTEXT_LENGTH", 8192)
        self.sample_interval = env_int("SAMPLE_INTERVAL", 10)
        self.startup_timeout = env_int("STARTUP_TIMEOUT", 1800)
        self.served_model_name = os.environ.get("SERVED_MODEL_NAME", self.model_path.name)
        self.sglang_bin = os.environ.get("SGLANG_BIN", "/usr/local/bin/sglang")
        self.modelscope_cli = Path(os.environ.get("MODELSCOPE_CLI", DEFAULT_MODELSCOPE_CLI)).expanduser()
        self.modelscope_model_id = os.environ.get("MODELSCOPE_MODEL_ID", DEFAULT_MODELSCOPE_MODEL_ID).strip()
        self.modelscope_max_workers = env_int("MODELSCOPE_MAX_WORKERS", 4)
        self.prompt_file = os.environ.get("PROMPT_FILE", "").strip()

        self.server: subprocess.Popen[bytes] | None = None
        self.downloader: subprocess.Popen[bytes] | None = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.active_concurrency = self.initial_concurrency
        self.request_counter = itertools.count(1)
        self.completed = 0
        self.failed = 0
        self.prompts = self._load_prompts()
        self.endpoint = f"http://127.0.0.1:{self.port}"

        self._validate_config()

    def _validate_config(self) -> None:
        if self.model_path.exists() and not self.model_path.is_dir():
            raise ValueError(f"Model path exists but is not a directory: {self.model_path}")
        if not Path(self.sglang_bin).is_file():
            raise ValueError(f"SGLang command not found: {self.sglang_bin}")
        if not self.modelscope_model_id or "/" not in self.modelscope_model_id:
            raise ValueError("MODELSCOPE_MODEL_ID must use the owner/name format")
        if self.modelscope_max_workers < 1:
            raise ValueError("MODELSCOPE_MAX_WORKERS must be at least 1")
        if not 1 <= self.port <= 65535:
            raise ValueError("SGLANG_HTTP_PORT must be between 1 and 65535")
        if not 1 <= self.target_util <= 100:
            raise ValueError("TARGET_GPU_UTIL must be between 1 and 100")
        if self.hysteresis < 0:
            raise ValueError("UTIL_HYSTERESIS cannot be negative")
        if self.max_concurrency < 1:
            raise ValueError("MAX_CONCURRENCY must be at least 1")
        if not 1 <= self.initial_concurrency <= self.max_concurrency:
            raise ValueError("INITIAL_CONCURRENCY must be between 1 and MAX_CONCURRENCY")
        if self.max_tokens < 1 or self.context_length < 1:
            raise ValueError("MAX_TOKENS and SGLANG_CONTEXT_LENGTH must be positive")
        if self.sample_interval < 1 or self.startup_timeout < 1:
            raise ValueError("SAMPLE_INTERVAL and STARTUP_TIMEOUT must be positive")

    def _load_prompts(self) -> list[str]:
        if not self.prompt_file:
            logging.info("No PROMPT_FILE set; using built-in synthetic prompts.")
            return []

        prompt_path = Path(self.prompt_file).expanduser()
        if not prompt_path.is_file():
            raise ValueError(f"PROMPT_FILE does not exist: {prompt_path}")
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"PROMPT_FILE has no non-empty lines: {prompt_path}")
        logging.info("Loaded %d prompts from %s", len(prompts), prompt_path)
        return prompts

    def _server_command(self) -> list[str]:
        host = os.environ.get("SGLANG_HOST", "0.0.0.0")
        return [
            self.sglang_bin,
            "serve",
            "--model-path",
            str(self.model_path),
            "--tp-size",
            "2",
            "--host",
            host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model_name,
            "--context-length",
            str(self.context_length),
            "--max-running-requests",
            str(self.max_concurrency),
            "--watchdog-timeout",
            "3600",
            "--dist-timeout",
            "3600",
        ]

    def start_server(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        command = self._server_command()
        logging.info("Starting one SGLang model with TP=2: %s", " ".join(command))
        server_env = os.environ.copy()
        # SGLang itself uses SGLANG_PORT when allocating internal ports.
        # Reusing it for the HTTP endpoint makes an internal listener occupy
        # the same port before Uvicorn starts.
        server_env.pop("SGLANG_PORT", None)
        with SERVER_LOG.open("a", encoding="utf-8") as log_file:
            self.server = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=server_env,
            )
        logging.info("SGLang server process started (PID %s).", self.server.pid)

    def _terminate_downloader(self) -> None:
        if self.downloader is None or self.downloader.poll() is not None:
            return
        try:
            os.killpg(self.downloader.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.downloader.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logging.warning("ModelScope download did not stop within 30 seconds; sending SIGKILL.")
            try:
                os.killpg(self.downloader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.downloader.wait(timeout=10)

    def ensure_model_available(self) -> bool:
        if model_snapshot_is_complete(self.model_path):
            logging.info("Complete local model snapshot found: %s", self.model_path)
            return True
        if self.stop_event.is_set():
            return False
        if not self.modelscope_cli.is_file() or not os.access(self.modelscope_cli, os.X_OK):
            logging.error("Model files are missing and ModelScope CLI is unavailable: %s", self.modelscope_cli)
            return False

        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            logging.exception("Cannot create model directory: %s", self.model_path)
            return False

        tracker_path = self.model_path.parent / f".{self.model_path.name}.modelscope-download.json"
        command = [
            str(self.modelscope_cli),
            "download",
            self.modelscope_model_id,
            "--repo-type",
            "model",
            "--local-dir",
            str(self.model_path),
            "--max-workers",
            str(self.modelscope_max_workers),
            "--continue-on-error",
            "--tracker-path",
            str(tracker_path),
            "--retry-failed",
        ]
        logging.info(
            "Model snapshot is missing or incomplete. Downloading %s to %s with %d workers.",
            self.modelscope_model_id,
            self.model_path,
            self.modelscope_max_workers,
        )
        try:
            self.downloader = subprocess.Popen(command, start_new_session=True)
        except OSError:
            logging.exception("Could not start ModelScope downloader: %s", self.modelscope_cli)
            return False

        while self.downloader.poll() is None:
            if self.stop_event.wait(1):
                logging.info("Stopping ModelScope download because the runner is stopping.")
                self._terminate_downloader()
                return False

        return_code = self.downloader.wait()
        self.downloader = None
        if self.stop_event.is_set():
            return False
        if return_code != 0:
            logging.error("ModelScope download exited with code %d; model server will not start.", return_code)
            return False
        if not model_snapshot_is_complete(self.model_path):
            logging.error(
                "ModelScope download finished, but the model snapshot is still incomplete: %s",
                self.model_path,
            )
            return False

        logging.info("Model download completed and required files are present.")
        return True

    def wait_until_ready(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"{self.endpoint}/health"
        logging.info("Waiting for SGLang health endpoint: %s", health_url)

        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if self.server is not None and self.server.poll() is not None:
                logging.error("SGLang exited during startup with code %s.", self.server.returncode)
                return False
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        logging.info("SGLang is ready.")
                        return True
            except (OSError, urllib.error.URLError):
                pass
            self.stop_event.wait(5)

        if not self.stop_event.is_set():
            logging.error("Timed out after %d seconds waiting for model startup.", self.startup_timeout)
        return False

    def _gpu_utilization(self) -> list[float]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values: list[float] = []
        for line in result.stdout.splitlines():
            value = line.strip().split()[0] if line.strip() else ""
            try:
                values.append(float(value))
            except ValueError:
                continue
        if len(values) < 2:
            raise RuntimeError(f"Expected utilization for 2 PPUs, got: {result.stdout!r}")
        return values[:2]

    def _next_prompt(self) -> str:
        number = next(self.request_counter)
        if self.prompts:
            base = self.prompts[(number - 1) % len(self.prompts)]
        else:
            base = PROMPT_TEMPLATES[(number - 1) % len(PROMPT_TEMPLATES)]
        # A changing suffix avoids replaying an identical cached prefix forever.
        return f"{base}\n\n请针对本次请求编号 {number} 给出完整、具体的回答。"

    def _send_request(self) -> None:
        body = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": self._next_prompt()}],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            response.read()

    def _worker(self, worker_id: int) -> None:
        while not self.stop_event.is_set():
            with self.condition:
                while not self.stop_event.is_set() and worker_id >= self.active_concurrency:
                    self.condition.wait(timeout=2)
                if self.stop_event.is_set():
                    return
            try:
                self._send_request()
                with self.condition:
                    self.completed += 1
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                with self.condition:
                    self.failed += 1
                logging.warning("Request worker %d failed: %s", worker_id, exc)
                self.stop_event.wait(3)
            except Exception:
                with self.condition:
                    self.failed += 1
                logging.exception("Unexpected request error in worker %d.", worker_id)
                self.stop_event.wait(3)

    def _controller(self) -> None:
        while not self.stop_event.wait(self.sample_interval):
            try:
                values = self._gpu_utilization()
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                logging.warning("Could not read PPU utilization: %s", exc)
                continue

            average = sum(values) / len(values)
            with self.condition:
                if average < self.target_util - self.hysteresis:
                    self.active_concurrency = min(self.max_concurrency, self.active_concurrency + 1)
                elif average > self.target_util + self.hysteresis:
                    self.active_concurrency = max(1, self.active_concurrency - 1)
                self.condition.notify_all()
                active = self.active_concurrency
                completed = self.completed
                failed = self.failed

            logging.info(
                "PPU utilization=%s%% (avg %.1f%%), requests=%d/%d, completed=%d, failed=%d",
                ",".join(f"{value:.0f}" for value in values),
                average,
                active,
                self.max_concurrency,
                completed,
                failed,
            )

    def _terminate_server(self) -> None:
        if self.server is None or self.server.poll() is not None:
            return
        try:
            os.killpg(self.server.pid, signal.SIGTERM)
            self.server.wait(timeout=30)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            logging.warning("SGLang did not stop within 30 seconds; sending SIGKILL.")
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.server.wait(timeout=10)

    def run(self) -> int:
        workers: list[threading.Thread] = []
        controller: threading.Thread | None = None
        try:
            if not self.ensure_model_available():
                return 0 if self.stop_event.is_set() else 2
            if self.stop_event.is_set():
                return 0
            self.start_server()
            if not self.wait_until_ready():
                return 1

            workers = [
                threading.Thread(target=self._worker, args=(index,), name=f"request-{index}")
                for index in range(self.max_concurrency)
            ]
            for worker in workers:
                worker.start()
            controller = threading.Thread(target=self._controller, name="utilization-controller")
            controller.start()

            logging.info(
                "Load control active: target %.1f%% +/- %.1f%%, concurrency %d..%d.",
                self.target_util,
                self.hysteresis,
                self.initial_concurrency,
                self.max_concurrency,
            )
            while not self.stop_event.wait(1):
                if self.server is not None and self.server.poll() is not None:
                    logging.error("SGLang server exited with code %s.", self.server.returncode)
                    return 1
            return 0
        finally:
            self.stop_event.set()
            with self.condition:
                self.condition.notify_all()
            self._terminate_downloader()
            self._terminate_server()
            if controller is not None:
                controller.join(timeout=5)
            for worker in workers:
                worker.join(timeout=5)
            self._remove_pid_file()
            logging.info("Runner stopped.")

    @staticmethod
    def _remove_pid_file() -> None:
        try:
            if PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                PID_FILE.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) != 3 or sys.argv[1] != "--model-path":
        print(f"Usage: {sys.argv[0]} --model-path /path/to/model", file=sys.stderr)
        return 2

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        runner = SGLangRunner(Path(sys.argv[2]).expanduser().resolve())
    except (OSError, ValueError) as exc:
        logging.error("Invalid configuration: %s", exc)
        return 2

    def request_stop(_signum: int, _frame: object) -> None:
        logging.info("Stop signal received.")
        runner.stop_event.set()
        with runner.condition:
            runner.condition.notify_all()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
