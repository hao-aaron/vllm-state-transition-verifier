"""What each vLLM weight-transfer engine can serve, and how to start it.

``ENGINE_CAPS`` records what the engine itself supports today. The harness does
not work around a missing capability (for example by gathering pipeline stages
onto one sender by hand): that would time harness code, not the engine. When an
engine gains a capability, flip its flag and the layouts that need it run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from layout import TrainerLayout


@dataclass(frozen=True)
class EngineCaps:
    # Several trainer ranks send different data (one sender per PP stage, ...).
    multi_sender: bool
    # A rank may hold only part of the model (PP stages, EP groups).
    partial_ownership: bool
    # Trainer ranks must share GPUs with the vLLM workers.
    colocated_only: bool = False
    # Trainer mesh dimensions (of size > 1) within a stage the engine can describe.
    max_mesh_dims: int | None = None
    # Mesh dimensions one tensor may be sharded over (fsdp and tp both shard).
    max_sharded_axes: int | None = None
    # Distinct source shards per tensor.
    max_src_shards: int | None = None
    # The trainer ranks must be Ray actors (the engine spawns Ray actors itself).
    needs_ray: bool = False
    # The harness drives this engine (later milestones fill in the rest).
    implemented: bool = False


ENGINE_CAPS: dict[str, EngineCaps] = {
    "nccl": EngineCaps(multi_sender=False, partial_ownership=False, implemented=True),
    "ipc": EngineCaps(
        multi_sender=False,
        partial_ownership=False,
        colocated_only=True,
        implemented=True,
    ),
    "nccl_m2n": EngineCaps(
        multi_sender=True,
        partial_ownership=False,
        max_mesh_dims=2,
        max_sharded_axes=1,
        max_src_shards=16,
        implemented=True,
    ),
    "sharded_rdt": EngineCaps(
        multi_sender=True, partial_ownership=True, needs_ray=True, implemented=True
    ),
}


LAUNCHERS = ("torchrun", "ray")


def unsupported_reasons(
    layout: TrainerLayout, backend: str, launcher: str = "torchrun"
) -> list[str]:
    """Why ``backend`` cannot serve a trainer with ``layout`` started by
    ``launcher``; empty if it can."""
    caps = ENGINE_CAPS.get(backend)
    if caps is None:
        return [f"unknown backend {backend!r}; known: {sorted(ENGINE_CAPS)}"]
    reasons = []
    if not caps.implemented:
        reasons.append(f"{backend}: not driven by the harness yet")
    if launcher not in LAUNCHERS:
        reasons.append(f"unknown launcher {launcher!r}; known: {LAUNCHERS}")
    elif caps.needs_ray and launcher != "ray":
        reasons.append(
            f"{backend}: trainer ranks must run as Ray actors (--launcher ray)"
        )
    for dim in ("pp", "ep"):
        size = getattr(layout, dim)
        if size == 1:
            continue
        missing = [
            cap
            for cap in ("multi_sender", "partial_ownership")
            if not getattr(caps, cap)
        ]
        if missing:
            reasons.append(
                f"{backend}: {dim}={size} needs each {dim} group to send what it "
                f"holds, but the engine lacks {', '.join(missing)}"
            )
    if caps.max_mesh_dims is not None and layout.stage_mesh_dims > caps.max_mesh_dims:
        reasons.append(
            f"{backend}: {layout.stage_mesh_dims} mesh dims of size > 1, engine "
            f"supports at most {caps.max_mesh_dims}"
        )
    sharded_axes = (layout.fsdp > 1) + (layout.tp > 1)
    if caps.max_sharded_axes is not None and sharded_axes > caps.max_sharded_axes:
        reasons.append(
            f"{backend}: fsdp and tp both shard a tensor, engine shards over at "
            f"most {caps.max_sharded_axes} mesh axis"
        )
    shards = layout.fsdp * layout.tp
    if caps.max_src_shards is not None and shards > caps.max_src_shards:
        reasons.append(
            f"{backend}: {shards} source shards per tensor, engine supports at "
            f"most {caps.max_src_shards}"
        )
    return reasons


# Server environment each engine needs on top of VLLM_SERVER_DEV_MODE.
SERVER_ENV: dict[str, dict[str, str]] = {
    # IPC handles travel pickled over HTTP.
    "ipc": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
    # RDT pulls into RayExecutorV2 tensor-transport actors.
    "sharded_rdt": {"VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1"},
}
# vllm serve arguments each engine needs.
SERVER_ARGS: dict[str, list[str]] = {
    "sharded_rdt": ["--distributed-executor-backend", "ray"],
}


def server_transfer_config(backend: str) -> str:
    return json.dumps({"backend": backend})


def nccl_init_info(
    *, rank: int, master_address: str, master_port: int, server_world_size: int
) -> Any:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

    return NCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=server_world_size + 1,
        rank=rank,
        packed=True,
    )


def m2n_init_info(
    *,
    rank: int,
    master_address: str,
    master_port: int,
    layout: TrainerLayout,
    server_world_size: int,
    server_tp: int,
) -> Any:
    from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo

    return M2NTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=layout.world_size + server_world_size,
        num_trainer_ranks=layout.world_size,
        dst_mesh_dims=(server_world_size // server_tp, server_tp),
        rank=rank,
    )


def ipc_init_info(*, rank: int) -> Any:
    from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

    return IPCTrainerInitInfo(rank=rank, packed=True)


def rdt_init_info(*, rank: int, server_world_size: int) -> Any:
    import ray

    from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
        ShardedRDTTrainerInitInfo,
    )

    return ShardedRDTTrainerInitInfo(
        rank=rank,
        num_consumers=server_world_size,
        trainer_actor_namespace=ray.get_runtime_context().namespace,
    )
