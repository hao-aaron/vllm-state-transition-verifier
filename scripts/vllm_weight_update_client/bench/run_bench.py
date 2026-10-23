#!/usr/bin/env python3
"""Benchmark one weight-transfer engine under one trainer layout.

Starts a vLLM server on checkpoint A (or attaches to one with ``--base-url``),
runs ``trainer_main.py`` under torchrun to push the update checkpoints through
the engine, and writes ``result.json`` and ``report.md``; each run also appends
one row to ``<output-dir>/../results.jsonl``.

    run_bench.py --model-a <A> --update <B> --update <C> \
        --backend nccl --layout fsdp=2,tp=2 \
        --server-gpus 0,1 --trainer-gpus 2,3,4,5 --output-dir <dir> \
        -- --tensor-parallel-size 2 --max-model-len 4096

With ``--audit`` the run ends with the refit-equals-fresh tensor audit: the
server after the last sync must hold the bytes of a fresh server started on the
last checkpoint sent. Fresh snapshots are cached under ``--fresh-cache-dir``,
keyed by checkpoint files, server arguments and vLLM version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / "validation"
sys.path[:0] = [str(HERE), str(VALIDATION)]

import tensor_audit  # noqa: E402
from engines import (  # noqa: E402
    ENGINE_CAPS,
    LAUNCHERS,
    SERVER_ARGS,
    SERVER_ENV,
    server_transfer_config,
    unsupported_reasons,
)
from layout import TrainerLayout  # noqa: E402
from server import Server, get_json, warm  # noqa: E402


def _trainer_world_size(layout: str) -> int:
    return 1 if layout == "trainerless" else TrainerLayout.parse(layout).world_size


def _snapshot_pair(url: str, model: str, out: Path, name: str) -> tuple:
    snaps = []
    for suffix in ("", "_repeat"):
        warm(url, model)
        snap = tensor_audit.fetch_snapshot(url)
        (out / f"{name}{suffix}.json").write_text(json.dumps(snap) + "\n")
        snaps.append(snap)
    return tuple(snaps)


def _fresh_key(checkpoint: str, args: argparse.Namespace, version: str) -> str:
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    identity = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "files": [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files],
        "serve_args": args.serve_args,
        "server_gpus": len(args.server_gpus.split(",")),
        "vllm_version": version,
    }
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]


def _fresh_snapshots(args: argparse.Namespace, checkpoint: str, version: str) -> tuple:
    cache = args.fresh_cache_dir / _fresh_key(checkpoint, args, version)
    if (cache / "fresh_repeat.json").is_file():
        print(f"fresh snapshot from cache {cache}")
        return tuple(
            json.loads((cache / f"fresh{s}.json").read_text()) for s in ("", "_repeat")
        )
    cache.mkdir(parents=True, exist_ok=True)
    server = _start(args, checkpoint, args.output_dir / "server_fresh.log")
    try:
        return _snapshot_pair(server.url, server.model, cache, "fresh")
    finally:
        server.stop()


def _extra_env(args: argparse.Namespace) -> dict[str, str]:
    return dict(item.split("=", 1) for item in args.env)


def _start(args: argparse.Namespace, model: str, log: Path) -> Server:
    env = {**SERVER_ENV.get(args.backend, {}), **_extra_env(args)}
    if ENGINE_CAPS[args.backend].needs_ray:
        env["RAY_ADDRESS"] = args.ray_address
        # Ray workers get the driver's PYTHONPATH (the audit extension) too.
        env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(["PYTHONPATH", *env])
    return Server(
        model,
        log=log,
        port=args.port,
        gpus=args.server_gpus,
        python=args.python,
        weight_transfer_config=server_transfer_config(args.backend),
        serve_args=[*SERVER_ARGS.get(args.backend, []), *args.serve_args],
        audit=args.audit,
        env=env,
        startup_timeout=args.startup_timeout,
    )


class RayHead:
    """A private single-node Ray cluster for engines whose server workers and
    trainer ranks must share one cluster (sharded_rdt)."""

    def __init__(self, args: argparse.Namespace) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.address = f"127.0.0.1:{port}"
        gpus = sorted(
            {*args.server_gpus.split(","), *args.trainer_gpus.split(",")}, key=int
        )
        env = dict(
            os.environ,
            **_extra_env(args),
            CUDA_VISIBLE_DEVICES=",".join(gpus),
            PYTHONPATH=os.pathsep.join(
                [str(VALIDATION), str(HERE), str(HERE.parent)]
                + [os.environ.get("PYTHONPATH", "")]
            ),
        )
        ray = str(Path(args.python).parent / "ray")
        cmd = [
            ray,
            "start",
            "--head",
            "--block",
            f"--port={port}",
            "--node-ip-address=127.0.0.1",
            f"--num-gpus={len(gpus)}",
            "--include-dashboard=false",
            "--disable-usage-stats",
        ]
        if args.ray_temp_dir:
            cmd.append(f"--temp-dir={args.ray_temp_dir}")
        self.handle = (args.output_dir / "ray_head.log").open("w")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("ray head exited, see ray_head.log")
            status = subprocess.run(
                [ray, "status", f"--address={self.address}"],
                capture_output=True,
                check=False,
            )
            if status.returncode == 0:
                return
            time.sleep(2)
        self.stop()
        raise TimeoutError("ray head did not come up, see ray_head.log")

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.handle.close()


def _launcher_cmd(args: argparse.Namespace) -> list[str]:
    if args.launcher == "ray":
        cmd = [args.python, str(HERE / "ray_launcher.py")]
        cmd += ["--ray-strategy", args.ray_strategy]
        for item in args.env:
            cmd += ["--env", item]
        if args.ray_address:
            cmd += ["--ray-address", args.ray_address]
        elif args.ray_temp_dir:
            cmd += ["--ray-temp-dir", args.ray_temp_dir]
        return [*cmd, "--"]
    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(_trainer_world_size(args.layout)),
        str(HERE / "trainer_main.py"),
    ]


def run_trainer(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    out = args.output_dir / "result.json"
    cmd = [
        *_launcher_cmd(args),
        "--base-url",
        base_url,
        "--backend",
        args.backend,
        "--layout",
        args.layout,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--output",
        str(out),
    ]
    for checkpoint in args.update:
        cmd += ["--checkpoint", checkpoint]
    if args.placement_rules:
        cmd += ["--placement-rules", str(args.placement_rules)]
    cmd += ["--gather", args.gather, "--source-api", args.source_api]
    if args.audit:
        cmd.append("--server-memory")
    env = dict(os.environ, **_extra_env(args), CUDA_VISIBLE_DEVICES=args.trainer_gpus)
    log = args.output_dir / "trainer.log"
    with log.open("w") as handle:
        rc = subprocess.run(
            cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode
    if rc != 0:
        raise RuntimeError(f"trainer failed (exit {rc}), see {log}")
    return json.loads(out.read_text())


def render_report(result: dict[str, Any], audit: dict[str, Any] | None) -> str:
    s = result["summary"]
    lines = [
        f"## Weight-transfer benchmark: {result['backend']}, "
        f"trainer {result['trainer']['layout']} via {result['trainer']['launcher']}",
        "",
        f"- model bytes: {result['total_bytes'] / 2**30:.2f} GiB",
        f"- trainer ranks: {result['trainer']['world_size']}, "
        f"server ranks: {result['server_world_size']}",
        f"- server args: `{' '.join(result.get('serve_args', []))}`",
        f"- vLLM {result['vllm_version']}, torch {result['torch_version']}",
        "",
        "| syncs | wall median (s) | min | max | GiB/s | pause window (s) |",
        "|---|---|---|---|---|---|",
        f"| {s['iterations']} | {s['wall_seconds_median']:.3f} | "
        f"{s['wall_seconds_min']:.3f} | {s['wall_seconds_max']:.3f} | "
        f"{s['gib_per_second_median']:.2f} | "
        f"{s['pause_window_seconds_median']:.3f} |",
    ]
    resident = (audit or {}).get("steps", [{}])[0].get("engine_resident_bytes")
    if resident:
        lines += [
            "",
            "Engine-resident server memory after the first sync (per rank): "
            + ", ".join(f"{b / 2**20:.0f} MiB" for b in resident)
            + "; growth is checked from there.",
        ]
    if audit is not None:
        lines += ["", tensor_audit.render_markdown(audit)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s [options] -- [vllm serve args]",
    )
    parser.add_argument("--model-a", help="checkpoint the server starts on")
    parser.add_argument("--update", action="append", required=True)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--layout", default="", help='e.g. "fsdp=2" or "trainerless"')
    parser.add_argument("--placement-rules", type=Path)
    parser.add_argument(
        "--gather",
        choices=("layer", "tensor"),
        default="layer",
        help="trainer gather granularity (see trainer_main.py --gather)",
    )
    parser.add_argument(
        "--source-api",
        choices=("legacy", "declared"),
        default="legacy",
        help="WeightSource flavor (see trainer_main.py --source-api)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for the server and every trainer rank "
        "(e.g. LD_PRELOAD / VLLM_NCCL_SO_PATH for nccl_m2n); repeatable",
    )
    parser.add_argument("--launcher", choices=LAUNCHERS, default="torchrun")
    parser.add_argument(
        "--ray-address",
        help="Ray cluster for --launcher ray (default: a local one on --trainer-gpus)",
    )
    parser.add_argument("--ray-strategy", default="PACK")
    parser.add_argument(
        "--ray-temp-dir", help="temp dir of a Ray cluster this driver starts"
    )
    parser.add_argument("--server-gpus", default="0")
    parser.add_argument(
        "--trainer-gpus",
        default="",
        help="CUDA_VISIBLE_DEVICES of the trainer (ignored with --ray-address)",
    )
    parser.add_argument(
        "--base-url", help="attach to a running server instead of starting one"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--fresh-cache-dir", type=Path, default=Path("~/.cache/wu-bench").expanduser()
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--python", default=sys.executable)
    argv = sys.argv[1:]
    serve_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, serve_args = argv[:split], argv[split + 1 :]
    args = parser.parse_args(argv)
    args.serve_args = serve_args
    if args.layout != "trainerless":
        reasons = unsupported_reasons(
            TrainerLayout.parse(args.layout), args.backend, args.launcher
        )
        if reasons:
            raise SystemExit("unsupported:\n  " + "\n  ".join(reasons))
    if args.base_url and args.audit:
        parser.error("--audit needs a server this driver starts (no --base-url)")
    if not args.base_url and not args.model_a:
        parser.error("pass --model-a to start a server, or --base-url")
    trainer_gpus = args.trainer_gpus.split(",")
    if not args.ray_address and len(trainer_gpus) != _trainer_world_size(args.layout):
        parser.error(f"--trainer-gpus lists {len(trainer_gpus)} GPUs for {args.layout}")
    if not args.base_url and not args.ray_address:
        server_gpus = set(args.server_gpus.split(","))
        caps = ENGINE_CAPS[args.backend]
        if caps.colocated_only and set(trainer_gpus) != server_gpus:
            parser.error(f"{args.backend} needs --trainer-gpus == --server-gpus")
        if not caps.colocated_only and set(trainer_gpus) & server_gpus:
            parser.error(f"{args.backend} needs trainer and server on separate GPUs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ray_head = None
    if ENGINE_CAPS[args.backend].needs_ray and not args.ray_address:
        if args.base_url:
            parser.error(f"{args.backend} with --base-url needs --ray-address")
        ray_head = RayHead(args)
        args.ray_address = ray_head.address
    try:
        audit = None
        if args.base_url:
            result = run_trainer(args, args.base_url)
        else:
            server = _start(args, args.model_a, args.output_dir / "server.log")
            try:
                version = get_json(f"{server.url}/version").get("version")
                if args.audit:
                    before = _snapshot_pair(
                        server.url, server.model, args.output_dir, "before"
                    )
                result = run_trainer(args, server.url)
                if args.audit:
                    after = _snapshot_pair(
                        server.url, server.model, args.output_dir, "after"
                    )
            finally:
                server.stop()
            if args.audit:
                fresh = _fresh_snapshots(args, result["final_checkpoint"], version)
                runtime = {kind: set() for kind in tensor_audit.KINDS}
                for first, second in (before, after, fresh):
                    for kind, names in tensor_audit.runtime_names(
                        first, second
                    ).items():
                        runtime[kind] |= names
                # Leak check from after the first sync: engines may keep buffers
                # they allocate once (RDT receive rings); that is reported as
                # engine-resident memory, not counted as growth.
                first = result["iterations"][0]["server_memory"]
                step = tensor_audit.compare_step(
                    before[1], after[0], fresh[0], runtime, memory_before=first
                )
                step["engine_resident_bytes"] = [
                    m["allocated"] - b["memory"]["allocated"]
                    for m, b in zip(first, before[1], strict=True)
                ]
                step["name"] = f"after {len(result['iterations'])} syncs"
                audit = {"status": step["status"], "steps": [step]}
                result["audit"] = audit

    finally:
        if ray_head is not None:
            ray_head.stop()

    result["serve_args"] = serve_args
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    report = render_report(result, audit)
    (args.output_dir / "report.md").write_text(report)
    row = {
        "backend": result["backend"],
        "layout": result["trainer"]["layout"],
        "launcher": args.launcher,
        "trainer_world_size": result["trainer"]["world_size"],
        "server_world_size": result["server_world_size"],
        "serve_args": serve_args,
        "env": args.env,
        "total_bytes": result["total_bytes"],
        "vllm_version": result["vllm_version"],
        "audit": audit["status"] if audit else None,
        **result["summary"],
    }
    with (args.output_dir.parent / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(report)
    if audit is not None and audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
