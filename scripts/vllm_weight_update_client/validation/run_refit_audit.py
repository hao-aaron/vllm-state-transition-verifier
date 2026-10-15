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
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent
sys.path.insert(0, str(HERE))

import tensor_audit  # noqa: E402
from server import Server, get_json  # noqa: E402


def start_server(args: argparse.Namespace, model: str, log: Path) -> Server:
    return Server(
        model,
        log=log,
        port=args.port,
        gpus=args.server_gpus,
        python=args.python,
        weight_transfer_config=args.weight_transfer_config,
        serve_args=args.serve_args,
        startup_timeout=args.startup_timeout,
    )


def snapshot(server: Server, path: Path) -> tensor_audit.Snapshot:
    snap = tensor_audit.fetch_snapshot(server.url)
    path.write_text(json.dumps(snap) + "\n")
    return snap


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
        "--weight-transfer-config",
        default='{"backend":"nccl"}',
        help='server config; add "packed": true on vLLM builds that have it',
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
    server = start_server(args, args.model_a, out / "server_a.log")
    try:
        version = get_json(f"{server.url}/version").get("version")
        server.warm()
        snaps["a"] = snapshot(server, out / "a.json")
        server.warm()
        snaps["a_repeat"] = snapshot(server, out / "a_repeat.json")
        runtime_pairs.append(("a", "a_repeat"))
        for name, checkpoint in zip(names, args.update, strict=False):
            publish(args, server.url, checkpoint, out / f"{name}_publish")
            server.warm()
            snaps[name] = snapshot(server, out / f"{name}.json")
            server.warm()
            snaps[f"{name}_repeat"] = snapshot(server, out / f"{name}_repeat.json")
            runtime_pairs.append((name, f"{name}_repeat"))
    finally:
        server.stop()
    for name, checkpoint in zip(names, args.update, strict=False):
        fresh = start_server(args, checkpoint, out / f"server_{name}_fresh.log")
        try:
            fresh.warm()
            snaps[f"{name}_fresh"] = snapshot(fresh, out / f"{name}_fresh.json")
            fresh.warm()
            snaps[f"{name}_fresh_repeat"] = snapshot(
                fresh, out / f"{name}_fresh_repeat.json"
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
