"""Hold a safetensors checkpoint as DTensors, reading only each rank's slice."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

import torch
import torch.distributed as dist
from layout import (
    EXPERT_MESH_DIMS,
    MESH_DIMS,
    PlacementRules,
    TrainerLayout,
    ep_groups,
    pp_stages,
)
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from vllm.distributed.weight_transfer import ParamMeta, WeightSource
from vllm.distributed.weight_transfer import Replicate as SourceReplicate
from vllm.distributed.weight_transfer import Shard as SourceShard
from vllm.distributed.weight_transfer.base import _stack_key, layerwise_groups

from hf_checkpoint_nccl_publisher import (
    _SAFETENSORS_DTYPE_TO_TORCH,
    CheckpointManifest,
)


def _placements(spec: tuple[str, ...]) -> list[Placement]:
    return [Replicate() if s == "R" else Shard(int(s[1:])) for s in spec]


def _contiguous_stride(shape: list[int]) -> tuple[int, ...]:
    stride, acc = [], 1
    for size in reversed(shape):
        stride.append(acc)
        acc *= size
    return tuple(reversed(stride))


def _schema(manifest: CheckpointManifest) -> list[tuple]:
    return [(t.name, t.shape, t.safetensors_dtype) for t in manifest.tensors]


def same_schema(a: CheckpointManifest, b: CheckpointManifest) -> bool:
    return _schema(a) == _schema(b)


def build_meshes(
    layout: TrainerLayout, device_type: str
) -> tuple[DeviceMesh, DeviceMesh | None]:
    """The ``(pp, dp, fsdp, tp)`` mesh, and with ep the ``(pp, ep, edp, tp)``
    mesh over the same ranks that expert tensors live on."""
    mesh = init_device_mesh(device_type, layout.mesh_shape, mesh_dim_names=MESH_DIMS)
    if layout.ep == 1:
        return mesh, None
    expert_mesh = init_device_mesh(
        device_type, layout.expert_mesh_shape, mesh_dim_names=EXPERT_MESH_DIMS
    )
    return mesh, expert_mesh


class ShardedCheckpoint:
    """This rank's share of a checkpoint under ``layout``.

    Tensors of other pipeline stages, and with ep the experts of other ep
    groups, are not held; ``owners`` maps every tensor to the global ranks
    that hold it, the same on every rank. ``load`` refills the local
    shards in place from another checkpoint with the same schema, the way a
    training step changes the weights without moving them.
    """

    def __init__(
        self,
        manifest: CheckpointManifest,
        layout: TrainerLayout,
        mesh: DeviceMesh,
        rules: PlacementRules,
        device: torch.device,
        expert_mesh: DeviceMesh | None = None,
    ) -> None:
        self.manifest = manifest
        self.rank = dist.get_rank()
        stage = layout.pp_rank(self.rank)
        stage_mesh = mesh["dp", "fsdp", "tp"]
        names = [t.name for t in manifest.tensors]
        stage_of = pp_stages(names, layout.pp)
        group_of = ep_groups(names, layout.ep) if expert_mesh is not None else {}
        if expert_mesh is not None:
            ep_group = expert_mesh.get_coordinate()[1]
            group_mesh = expert_mesh["edp", "tp"]
        self.owners: dict[str, tuple[int, ...]] = {}
        for name in names:
            if name in group_of:
                ranks = expert_mesh.mesh[stage_of[name], group_of[name]]
            else:
                ranks = mesh.mesh[stage_of[name]]
            self.owners[name] = tuple(ranks.flatten().tolist())
        self.tensors: dict[str, DTensor] = {}
        self._slices: dict[str, tuple[slice, ...]] = {}
        for meta in manifest.tensors:
            if stage_of[meta.name] != stage:
                continue
            spec = rules.placement_spec(meta.name, len(meta.shape))
            if meta.name in group_of:
                if group_of[meta.name] != ep_group:
                    continue
                tensor_mesh, placements = group_mesh, _placements(spec[1:])
            else:
                tensor_mesh, placements = stage_mesh, _placements(spec)
            local_shape, offset = compute_local_shape_and_global_offset(
                meta.shape, tensor_mesh, placements
            )
            local = torch.empty(
                local_shape,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[meta.safetensors_dtype][0],
                device=device,
            )
            self.tensors[meta.name] = DTensor.from_local(
                local,
                tensor_mesh,
                placements,
                run_check=False,
                shape=torch.Size(meta.shape),
                stride=_contiguous_stride(meta.shape),
            )
            self._slices[meta.name] = tuple(
                slice(o, o + n) for o, n in zip(offset, local_shape, strict=True)
            )
        self.load(manifest)

    @property
    def local_bytes(self) -> int:
        return sum(
            t.to_local().numel() * t.dtype.itemsize for t in self.tensors.values()
        )

    def load(self, manifest: CheckpointManifest) -> None:
        if not same_schema(manifest, self.manifest):
            raise ValueError(
                f"{manifest.checkpoint_path} does not have the schema of "
                f"{self.manifest.checkpoint_path}"
            )
        root = Path(manifest.checkpoint_path)
        by_file: dict[str, list[str]] = {}
        for meta in manifest.tensors:
            if meta.name in self.tensors:
                by_file.setdefault(meta.file, []).append(meta.name)
        for file_name, names in by_file.items():
            with safe_open(root / file_name, framework="pt", device="cpu") as f:
                for name in names:
                    local = self.tensors[name].to_local()
                    if local.numel() == 0:
                        continue
                    index = self._slices[name]
                    data = f.get_slice(name)[index] if index else f.get_tensor(name)
                    local.copy_(data.view(local.shape), non_blocking=False)
        self.manifest = manifest


def _chunk_len(length: int, n: int, index: int) -> int:
    size = -(-length // n)
    return max(0, min(size, length - index * size))


def gather_full(tensors: list[DTensor]) -> list[torch.Tensor]:
    """Full tensors of ``tensors``, gathered the way FSDP2 unshards a module:
    per mesh and sharded mesh dim, every tensor's shard is packed into one
    buffer and all-gathered in one collective, instead of a ``full_tensor()``
    each.

    The ranks of each mesh dim's group must pass the same tensors of that mesh
    in the same order, which holds for one layer's names.
    """
    full: list[torch.Tensor] = [t.to_local() for t in tensors]
    by_mesh: dict[tuple[str, ...], list[int]] = {}
    for i, t in enumerate(tensors):
        by_mesh.setdefault(t.device_mesh.mesh_dim_names, []).append(i)
    # one mesh order on every rank, so ranks in two meshes issue collectives alike
    for dims in sorted(by_mesh):
        idx = by_mesh[dims]
        mesh = tensors[idx[0]].device_mesh
        coord = mesh.get_coordinate()
        for d in reversed(range(mesh.ndim)):  # undo the innermost split first
            n = mesh.size(d)
            if n == 1:
                continue
            plan = []
            for i in idx:
                placement = tensors[i].placements[d]
                if not isinstance(placement, Shard):
                    continue
                k = placement.dim
                length = tensors[i].shape[k]
                for e in range(d):
                    outer = tensors[i].placements[e]
                    if isinstance(outer, Shard) and outer.dim == k:
                        length = _chunk_len(length, mesh.size(e), coord[e])
                rows = full[i].movedim(k, 0)
                pad = -(-length // n)
                if rows.shape[0] == pad:
                    padded = rows.contiguous()
                else:
                    padded = rows.new_zeros((pad, *rows.shape[1:]))
                    padded[: rows.shape[0]] = rows
                plan.append((i, k, length, padded))
            if not plan:
                continue
            sizes = [p[3].numel() * p[3].element_size() for p in plan]
            offsets = list(itertools.accumulate((-(-s // 16) * 16 for s in sizes)))
            buf = torch.empty(offsets[-1], dtype=torch.uint8, device=full[0].device)
            offsets = [0, *offsets[:-1]]
            for (_, _, _, padded), off, size in zip(plan, offsets, sizes):
                buf[off : off + size] = padded.reshape(-1).view(torch.uint8)
            out = buf.new_empty(n * buf.numel())
            dist.all_gather_into_tensor(out, buf, group=mesh.get_group(d))
            out = out.view(n, -1)
            for (i, k, length, padded), off, size in zip(plan, offsets, sizes):
                pieces = [
                    out[r, off : off + size]
                    .view(padded.dtype)
                    .view(padded.shape)[: _chunk_len(length, n, r)]
                    for r in range(n)
                ]
                full[i] = torch.cat(pieces).movedim(0, k)
    return [t.contiguous() for t in full]


class GatherSource(WeightSource):
    """Full tensors, gathered one layer at a time.

    Without ``partial`` every rank holds every tensor and iterates in manifest
    order. With ``partial`` (engines with partial ownership, e.g. a pp stage
    holding its layers), metadata is the manifest in ``layerwise_groups``
    order, so each gather group is contiguous, and declares each tensor's
    owners; a rank iterates only the names it owns.

    ``batched`` gathers each layer with ``gather_full``, as an FSDP2 trainer
    unshards a module; without it each tensor is gathered by its own
    ``full_tensor()``.
    """

    def __init__(
        self, checkpoint: ShardedCheckpoint, partial: bool, batched: bool = True
    ) -> None:
        self.checkpoint = checkpoint
        self.partial = partial
        self.batched = batched
        if not partial and len(checkpoint.tensors) != len(
            checkpoint.manifest.tensors
        ):
            raise ValueError("a full-ownership source must hold every tensor")

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        metadata = {
            t.name: ParamMeta(
                t.name,
                _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
                tuple(t.shape),
                owners=self.checkpoint.owners[t.name] if self.partial else None,
            )
            for t in self.checkpoint.manifest.tensors
        }
        if not self.partial:
            return list(metadata.values())
        groups = layerwise_groups(list(metadata))
        return [metadata[name] for g in groups for name in g]

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        names = self.owned_names(self.checkpoint.rank)
        if not self.batched:
            for name in names:
                yield name, self.checkpoint.tensors[name].full_tensor()
            return
        for _, layer in itertools.groupby(names, key=_stack_key):
            layer = list(layer)
            full = gather_full([self.checkpoint.tensors[n] for n in layer])
            ready = None
            if full[0].is_cuda:
                ready = torch.cuda.Event()
                ready.record()
            for name, tensor in zip(layer, full):
                # the layer was gathered on the stream current at its first
                # next(); a consumer may resume us on another (the packed nccl
                # sender switches streams per chunk)
                if ready is not None:
                    stream = torch.cuda.current_stream()
                    stream.wait_event(ready)
                    tensor.record_stream(stream)
                yield name, tensor


class ShardedM2NSource(WeightSource):
    """Local shards and their placements, for nccl_m2n (no gather).

    m2n meshes have at most two axes, so the stage mesh's size-1 dimensions are
    dropped; ``engines.unsupported_reasons`` has already refused layouts that
    need more.
    """

    def __init__(
        self,
        checkpoint: ShardedCheckpoint,
        layout: TrainerLayout,
        rules: PlacementRules,
    ) -> None:
        self.checkpoint = checkpoint
        self.layout = layout
        self.rules = rules

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        sizes = (self.layout.dp, self.layout.fsdp, self.layout.tp)
        axes = [i for i, size in enumerate(sizes) if size > 1] or [0]
        if len(axes) > 2:
            raise ValueError(f"nccl_m2n cannot describe layout {self.layout}")
        owners = tuple(range(self.layout.world_size))
        metadata = []
        for t in self.checkpoint.manifest.tensors:
            spec = self.rules.placement_spec(t.name, len(t.shape))
            metadata.append(
                ParamMeta(
                    t.name,
                    _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
                    tuple(t.shape),
                    owners=owners,
                    mesh_shape=tuple(sizes[i] for i in axes),
                    placements=tuple(
                        SourceReplicate()
                        if spec[i] == "R"
                        else SourceShard(int(spec[i][1:]))
                        for i in axes
                    ),
                )
            )
        return metadata

    def iter_local(self) -> Iterator[tuple[str, torch.Tensor]]:
        for meta in self.metadata:
            yield meta.name, self.checkpoint.tensors[meta.name].to_local()
