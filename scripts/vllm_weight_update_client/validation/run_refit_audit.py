#!/usr/bin/env python3
"""Refit-equals-fresh audit of NCCL weight updates on a single node.

Runs the protocol whose report backs a "reload-safe" claim:

1. serve checkpoint A with the tensor audit extension;
2. for each update checkpoint (B, C, ...): publish it through the NCCL
   publisher (pause / reset prefix cache / publish / resume) and snapshot every
   device tensor;
3. serve each update checkpoint fresh and snapshot it;
4. per update: the updated server must equal the fresh server byte for byte
   (registered and off-registry tensors), nothing may move or leak, and
   tensors the update did not change are listed as untested.

Use at least two different updates (A -> B -> C): some bugs only show on the
second update that changes values, after kernels were built once. Make the
update checkpoints with ``make_perturbed_checkpoint.py`` so that every tensor,
scales included, changes.

    run_refit_audit.py --model-a <A> --update <B> --update <C> \
        --server-gpus 0 --publisher-gpu 1 --output-dir <dir> -- <vllm serve args>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent
sys.path.insert(0, str(HERE))

import tensor_audit  # noqa: E402

EXTENSION = "tensor_audit_extension.TensorAuditExtension"
WARMUP_PROMPTS = (
    "State the capital of France in one word.",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
)


def _get(url: str, timeout: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def _post(url: str, payload: dict, timeout: float = 600) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Server:
    """A vLLM OpenAI server with the audit extension, in its own process group."""

    def __init__(self, args: argparse.Namespace, model: str, log: Path) -> None:
        self.url = f"http://127.0.0.1:{args.port}"
        env = dict(os.environ)
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = args.server_gpus
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(HERE), env.get("PYTHONPATH", "")) if p
        )
        cmd = [
            args.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(args.port),
            "--weight-transfer-config",
            args.weight_transfer_config,
            "--worker-extension-cls",
            EXTENSION,
            *args.serve_args,
        ]
        self.log = log
        self.handle = log.open("w")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup, see {log}")
            try:
                _get(f"{self.url}/health")
                self.model = _get(f"{self.url}/v1/models")["data"][0]["id"]
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)
        self.stop()
        raise TimeoutError(f"server did not become healthy, see {log}")

    def warm(self) -> None:
        """Requests between snapshots: build lazy caches and dirty runtime
        workspaces, so they show up as runtime state, not as weights."""
        for prompt in WARMUP_PROMPTS:
            try:
                _post(
                    f"{self.url}/v1/completions",
                    {
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                )
            except urllib.error.HTTPError as exc:  # e.g. longer than max_model_len
                print(f"warm-up request rejected ({exc.code}), continuing")

    def snapshot(self, path: Path) -> tensor_audit.Snapshot:
        snap = tensor_audit.fetch_snapshot(self.url)
        path.write_text(json.dumps(snap) + "\n")
        return snap

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.handle.close()


def publish(args: argparse.Namespace, url: str, checkpoint: str, out: Path) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.publisher_gpu
    cmd = [
        "bash",
        str(CLIENT / "run_vllm_lifecycle.sh"),
        url,
        "--",
        args.python,
        str(CLIENT / "run_vllm_weight_update.py"),
        "--base-url",
        url,
        "--model",
        checkpoint,
        "--revision",
        "local",
        "--checkpoint-path",
        checkpoint,
        "--device",
        "cuda:0",
        "--output",
        str(out.with_suffix(".json")),
    ]
    with out.with_suffix(".log").open("w") as log:
        rc = subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        ).returncode
    if rc != 0:
        raise RuntimeError(
            f"publishing {checkpoint} failed, see {out.with_suffix('.log')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s [options] -- [vllm serve args]",
    )
    parser.add_argument("--model-a", required=True, help="model the server starts on")
    parser.add_argument(
        "--update",
        action="append",
        required=True,
        help="local checkpoint to publish, in order; repeat (A -> B -> C)",
    )
    parser.add_argument("--server-gpus", required=True, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--publisher-gpu", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weight-transfer-config", default='{"backend":"nccl","packed":true}'
    )
    parser.add_argument("--memory-growth-limit-mb", type=float, default=64)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--python", default=sys.executable)
    argv = sys.argv[1:]
    serve_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, serve_args = argv[:split], argv[split + 1 :]
    args = parser.parse_args(argv)
    args.serve_args = serve_args
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    names = [f"update{i + 1}" for i in range(len(args.update))]
    snaps: dict[str, tensor_audit.Snapshot] = {}
    runtime_pairs: list[tuple[str, str]] = []
    server = Server(args, args.model_a, out / "server_a.log")
    try:
        version = _get(f"{server.url}/version").get("version")
        server.warm()
        snaps["a"] = server.snapshot(out / "a.json")
        server.warm()
        snaps["a_repeat"] = server.snapshot(out / "a_repeat.json")
        runtime_pairs.append(("a", "a_repeat"))
        for name, checkpoint in zip(names, args.update, strict=False):
            publish(args, server.url, checkpoint, out / f"{name}_publish")
            server.warm()
            snaps[name] = server.snapshot(out / f"{name}.json")
            server.warm()
            snaps[f"{name}_repeat"] = server.snapshot(out / f"{name}_repeat.json")
            runtime_pairs.append((name, f"{name}_repeat"))
    finally:
        server.stop()
    for name, checkpoint in zip(names, args.update, strict=False):
        fresh = Server(args, checkpoint, out / f"server_{name}_fresh.log")
        try:
            fresh.warm()
            snaps[f"{name}_fresh"] = fresh.snapshot(out / f"{name}_fresh.json")
            fresh.warm()
            snaps[f"{name}_fresh_repeat"] = fresh.snapshot(
                out / f"{name}_fresh_repeat.json"
            )
            runtime_pairs.append((f"{name}_fresh", f"{name}_fresh_repeat"))
        finally:
            fresh.stop()

    runtime = {kind: set() for kind in tensor_audit.KINDS}
    for first, second in runtime_pairs:
        for kind, found in tensor_audit.runtime_names(
            snaps[first], snaps[second]
        ).items():
            runtime[kind] |= found
    steps = []
    before = "a_repeat"
    for name, checkpoint in zip(names, args.update, strict=False):
        step = tensor_audit.compare_step(
            snaps[before],
            snaps[name],
            snaps[f"{name}_fresh"],
            runtime,
            int(args.memory_growth_limit_mb * 2**20),
        )
        step["name"] = f"{name}: {checkpoint}"
        steps.append(step)
        before = f"{name}_repeat"
    report = {
        "status": "PASS" if all(s["status"] == "PASS" for s in steps) else "FAIL",
        "model": args.model_a,
        "updates": args.update,
        "server_args": " ".join(serve_args),
        "transport": args.weight_transfer_config,
        "vllm_version": version,
        "steps": steps,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = tensor_audit.render_markdown(report)
    (out / "report.md").write_text(markdown)
    print(markdown)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
