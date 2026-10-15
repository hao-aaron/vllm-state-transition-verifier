#!/usr/bin/env python3
"""Benchmark trainer: push checkpoints into a running vLLM server.

Run it under torchrun with one process per trainer rank (or through
``ray_launcher.py``, which runs the same ranks as Ray actors):

    torchrun --nproc-per-node 4 trainer_main.py --base-url http://host:8000 \
        --backend nccl --layout fsdp=2,tp=2 \
        --checkpoint <B> --checkpoint <C> --output result.json

Each rank holds its share of the checkpoint as DTensors (see ``layout.py``).
Every sync goes through the engine's own trainer API, so the timing is the
engine's cost for that trainer layout. Syncs alternate between the given
checkpoints so the bytes really change; between syncs the trainer reloads its
shards in place, untimed, like a training step.

``--layout trainerless`` runs the single-process checkpoint publisher instead,
as the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "validation")]

import requests  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from engines import (  # noqa: E402
    ENGINE_CAPS,
    ipc_init_info,
    m2n_init_info,
    nccl_init_info,
    rdt_init_info,
    unsupported_reasons,
)
from layout import PlacementRules, TrainerLayout  # noqa: E402
from sharded_checkpoint import (  # noqa: E402
    GatherSource,
    ShardedCheckpoint,
    ShardedM2NSource,
    build_meshes,
)
from vllm.distributed.weight_transfer import (  # noqa: E402
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.utils.network_utils import get_ip, get_open_port  # noqa: E402

from hf_checkpoint_nccl_publisher import (  # noqa: E402
    CheckpointManifest,
    NcclCheckpointPublisher,
)


class TimedClient(HTTPVLLMWeightSyncClient):
    """Records the wall time of every weight-transfer RPC."""

    def __init__(self, base_url: str, timeout: float) -> None:
        super().__init__(base_url, timeout=timeout)
        self.calls: list[tuple[str, float]] = []

    def _post(self, path: str, json: dict[str, Any] | None = None) -> None:
        started = time.perf_counter()
        try:
            super()._post(path, json)
        finally:
            self.calls.append((path, time.perf_counter() - started))


class Lifecycle:
    """Pause / reset prefix cache / resume around a sync, as
    ``run_vllm_lifecycle.sh`` does."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str) -> dict:
        response = requests.post(
            f"{self.base_url}/{path}", json={}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def pause(self) -> None:
        self._post("pause?mode=keep&clear_cache=false")
        reply = self._post("reset_prefix_cache?reset_running_requests=false")
        if reply.get("success") is False:
            raise RuntimeError(f"reset_prefix_cache failed: {reply}")

    def resume(self) -> None:
        self._post("resume")


def _server_memory(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """Per-rank allocator stats of the server (needs the audit extension)."""
    if not args.server_memory:
        return None
    import tensor_audit

    snapshot = tensor_audit.fetch_snapshot(args.base_url, digests=False)
    return [rank["memory"] for rank in snapshot]


def _load_manifests(paths: list[str]) -> list[CheckpointManifest]:
    return [
        CheckpointManifest.load(model=p, revision="local", checkpoint_path=p)
        for p in paths
    ]


def _summary(records: list[dict], total_bytes: int) -> dict[str, Any]:
    walls = [r["wall_seconds"] for r in records if not r["warmup"]]
    if not walls:
        return {}
    median = statistics.median(walls)
    return {
        "iterations": len(walls),
        "wall_seconds_median": median,
        "wall_seconds_min": min(walls),
        "wall_seconds_max": max(walls),
        "gib_per_second_median": total_bytes / 2**30 / median,
        "pause_window_seconds_median": statistics.median(
            r["pause_window_seconds"] for r in records if not r["warmup"]
        ),
    }


def run_trainerless(args: argparse.Namespace, manifests: list) -> dict[str, Any]:
    lifecycle = Lifecycle(args.base_url, args.timeout)
    publisher = NcclCheckpointPublisher(
        base_url=args.base_url,
        manifest=manifests[0],
        device=f"cuda:{torch.cuda.current_device()}",
        timeout_seconds=args.timeout,
    )
    initialization = publisher.initialize()
    records = []
    try:
        for i in range(args.warmup + args.iters):
            manifest = manifests[i % len(manifests)]
            paused = time.perf_counter()
            lifecycle.pause()
            started = time.perf_counter()
            publisher.publish_update(target="main", manifest=manifest)
            wall = time.perf_counter() - started
            lifecycle.resume()
            pause_window = time.perf_counter() - paused
            records.append(
                {
                    "iteration": i,
                    "warmup": i < args.warmup,
                    "checkpoint": manifest.checkpoint_path,
                    "wall_seconds": wall,
                    "pause_window_seconds": pause_window,
                    "server_memory": _server_memory(args),
                }
            )
    finally:
        publisher.shutdown()
    return {
        "trainer": {"layout": "trainerless", "world_size": 1},
        "server_world_size": initialization["inference_world_size"],
        "iterations": records,
    }


def run_sharded(
    args: argparse.Namespace, layout: TrainerLayout, manifests: list
) -> dict[str, Any] | None:
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != layout.world_size:
        raise ValueError(
            f"layout {layout} needs {layout.world_size} ranks, "
            f"torchrun started {dist.get_world_size()}"
        )
    mesh, expert_mesh = build_meshes(layout, "cuda")
    started = time.perf_counter()
    rules = PlacementRules.load(args.placement_rules)
    checkpoint = ShardedCheckpoint(
        manifests[0], layout, mesh, rules, device, expert_mesh
    )
    load_seconds = time.perf_counter() - started
    client = TimedClient(args.base_url, args.timeout)
    lifecycle = Lifecycle(args.base_url, args.timeout)

    rendezvous: list[Any] = [None]
    if rank == 0:
        url = args.base_url.rstrip("/")
        server_world, server_tp = (
            requests.get(
                f"{url}/get_world_size?include_dp={include_dp}", timeout=args.timeout
            ).json()["world_size"]
            for include_dp in ("true", "false")
        )
        address = args.nccl_master_address or get_ip()
        rendezvous = [(address, get_open_port(), server_world, server_tp)]
    dist.broadcast_object_list(rendezvous, src=0)
    address, port, server_world, server_tp = rendezvous[0]
    endpoint = {"rank": rank, "master_address": address, "master_port": port}
    if args.backend == "nccl_m2n":
        source = ShardedM2NSource(checkpoint, layout, rules)
        init_info = m2n_init_info(
            **endpoint,
            layout=layout,
            server_world_size=server_world,
            server_tp=server_tp,
        )
    else:
        source = GatherSource(
            checkpoint, partial=ENGINE_CAPS[args.backend].partial_ownership
        )
        if args.backend == "ipc":
            init_info = ipc_init_info(rank=rank)
        elif args.backend == "sharded_rdt":
            init_info = rdt_init_info(rank=rank, server_world_size=server_world)
        else:
            init_info = nccl_init_info(**endpoint, server_world_size=server_world)
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info, client=client, source=source
    )

    records = []
    for i in range(args.warmup + args.iters):
        manifest = manifests[i % len(manifests)]
        if manifest is not checkpoint.manifest:
            checkpoint.load(manifest)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        dist.barrier()
        if rank == 0:
            client.calls.clear()
            paused = time.perf_counter()
            lifecycle.pause()
        dist.barrier()
        started = time.perf_counter()
        engine.send_weights()
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        dist.barrier()
        rank_record = {
            "rank": rank,
            "send_seconds": wall,
            "peak_transient_bytes": torch.cuda.max_memory_allocated() - allocated,
        }
        if rank == 0:
            lifecycle.resume()
            pause_window = time.perf_counter() - paused
        ranks: list[Any] | None = [None] * layout.world_size if rank == 0 else None
        dist.gather_object(rank_record, ranks, dst=0)
        if rank == 0:
            records.append(
                {
                    "iteration": i,
                    "warmup": i < args.warmup,
                    "checkpoint": manifest.checkpoint_path,
                    "wall_seconds": wall,
                    "pause_window_seconds": pause_window,
                    "server_memory": _server_memory(args),
                    "rpc_seconds": [list(call) for call in client.calls],
                    "ranks": ranks,
                }
            )

    local_bytes: list[Any] | None = [None] * layout.world_size if rank == 0 else None
    dist.gather_object(checkpoint.local_bytes, local_bytes, dst=0)
    engine.shutdown()
    dist.destroy_process_group()
    if rank != 0:
        return None
    return {
        "trainer": {
            "layout": str(layout),
            "world_size": layout.world_size,
            "load_seconds": load_seconds,
            "local_bytes_per_rank": local_bytes,
        },
        "server_world_size": server_world,
        "iterations": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument(
        "--layout",
        default="",
        help='e.g. "fsdp=2,tp=2" (dims pp, dp, fsdp, tp, ep), or "trainerless"',
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="local checkpoint to send; repeat to alternate between several",
    )
    parser.add_argument("--placement-rules", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--nccl-master-address",
        help="address the server reaches trainer rank 0 at (default: get_ip())",
    )
    parser.add_argument(
        "--server-memory",
        action="store_true",
        help="record server allocator stats after every sync (audit extension)",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, launcher: str) -> dict[str, Any] | None:
    """One trainer rank; rank 0 returns the result, the others None.

    Expects the torch.distributed environment (``RANK``, ``WORLD_SIZE``,
    ``MASTER_ADDR``, ``MASTER_PORT``, ``LOCAL_RANK``) from the launcher.
    """
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    manifests = _load_manifests(args.checkpoint)
    if args.layout == "trainerless":
        if args.backend != "nccl":
            raise ValueError("--layout trainerless only publishes over nccl")
        result = run_trainerless(args, manifests)
    else:
        layout = TrainerLayout.parse(args.layout)
        reasons = unsupported_reasons(layout, args.backend, launcher)
        if reasons:
            raise ValueError("unsupported:\n  " + "\n  ".join(reasons))
        result = run_sharded(args, layout, manifests)
    if result is None:
        return None

    import vllm

    result["trainer"]["launcher"] = launcher
    return {
        "backend": args.backend,
        "checkpoints": args.checkpoint,
        "final_checkpoint": result["iterations"][-1]["checkpoint"],
        "total_bytes": manifests[0].total_bytes,
        "manifest_sha256": manifests[0].manifest_sha256,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        **result,
        "summary": _summary(result["iterations"], manifests[0].total_bytes),
    }


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": result["summary"]}, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    result = run(args, launcher="torchrun")
    if result is not None:
        write_result(result, args.output)


if __name__ == "__main__":
    main()
