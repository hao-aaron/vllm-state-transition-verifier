"""A local vLLM OpenAI server for validation and benchmark drivers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AUDIT_EXTENSION = "tensor_audit_extension.TensorAuditExtension"
WARMUP_PROMPTS = (
    "State the capital of France in one word.",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
)


def get_json(url: str, timeout: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def post_json(url: str, payload: dict | None = None, timeout: float = 600) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


class Server:
    """A vLLM OpenAI server in its own process group.

    With ``audit=True`` the tensor audit extension is loaded, so the server can
    be snapshotted with ``tensor_audit.fetch_snapshot``.
    """

    def __init__(
        self,
        model: str,
        *,
        log: Path,
        port: int,
        gpus: str,
        python: str,
        weight_transfer_config: str,
        serve_args: list[str],
        audit: bool = True,
        env: dict[str, str] | None = None,
        startup_timeout: float = 1800,
    ) -> None:
        self.url = f"http://127.0.0.1:{port}"
        full_env = dict(os.environ)
        full_env["VLLM_SERVER_DEV_MODE"] = "1"
        full_env["CUDA_VISIBLE_DEVICES"] = gpus
        full_env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(HERE), full_env.get("PYTHONPATH", "")) if p
        )
        full_env.update(env or {})
        cmd = [
            python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--weight-transfer-config",
            weight_transfer_config,
            *(["--worker-extension-cls", AUDIT_EXTENSION] if audit else []),
            *serve_args,
        ]
        self.log = log
        self.handle = log.open("w")
        self.proc = subprocess.Popen(
            cmd,
            env=full_env,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.handle.close()
                raise RuntimeError(f"server exited during startup, see {log}")
            try:
                get_json(f"{self.url}/health")
                self.model = get_json(f"{self.url}/v1/models")["data"][0]["id"]
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)
        self.stop()
        raise TimeoutError(f"server did not become healthy, see {log}")

    def warm(self) -> None:
        """Requests between snapshots: build lazy caches and dirty runtime
        workspaces, so they show up as runtime state, not as weights."""
        warm(self.url, self.model)

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.handle.close()


def warm(url: str, model: str) -> None:
    for prompt in WARMUP_PROMPTS:
        try:
            post_json(
                f"{url}/v1/completions",
                {"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0},
            )
        except urllib.error.HTTPError as exc:  # e.g. longer than max_model_len
            print(f"warm-up request rejected ({exc.code}), continuing")
