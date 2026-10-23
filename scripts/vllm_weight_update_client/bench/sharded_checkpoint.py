"""Hold a safetensors checkpoint as DTensors, reading only each rank's slice."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
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

try:  # the declared sources need vLLM with ParamMeta ownership (#63215)
    from vllm.distributed.weight_transfer import Replicate as VReplicate
    from vllm.distributed.weight_transfer import Shard as VShard
except ImportError:
    VReplicate = VShard = None
from vllm.distributed.weight_transfer.base import _stack_key, layerwise_groups
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
)

try:  # legacy M2N source API, removed by vLLM #63215
    from vllm.distributed.weight_transfer.m2n_source import M2NWeightSource
except ImportError:
    M2NWeightSource = WeightSource

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
    groups, are not held. ``load`` refills the local
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
        self.layout = layout
        self.rules = rules
        self.mesh = mesh
        self.expert_mesh = expert_mesh
        stage = layout.pp_rank(dist.get_rank())
        stage_mesh = mesh["dp", "fsdp", "tp"]
        names = [t.name for t in manifest.tensors]
        stage_of = pp_stages(names, layout.pp)
        group_of = ep_groups(names, layout.ep) if expert_mesh is not None else {}
        if expert_mesh is not None:
            ep_group = expert_mesh.get_coordinate()[1]
            group_mesh = expert_mesh["edp", "tp"]
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
    order, so each gather group is contiguous; a rank iterates only the groups
    it holds part of, yielding ``None`` for names held elsewhere.

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
        metadata = {
            t.name: ParamMeta(
                t.name,
                _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
                tuple(t.shape),
            )
            for t in checkpoint.manifest.tensors
        }
        if partial:
            groups = layerwise_groups(list(metadata))
            self._metadata = [metadata[name] for g in groups for name in g]
        else:
            if len(checkpoint.tensors) != len(metadata):
                raise ValueError("a full-ownership source must hold every tensor")
            self._metadata = list(metadata.values())

    def metadata(self) -> list[ParamMeta]:
        return self._metadata

    def held_names(self) -> set[str] | None:
        return set(self.checkpoint.tensors) if self.partial else None

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor | None]]:
        names = (
            [name for g in self.groups() for name in g]
            if self.partial
            else [m.name for m in self._metadata]
        )
        if not self.batched:
            for name in names:
                tensor = self.checkpoint.tensors.get(name)
                yield name, None if tensor is None else tensor.full_tensor()
            return
        for _, layer in itertools.groupby(names, key=_stack_key):
            layer = list(layer)
            held = [n for n in layer if n in self.checkpoint.tensors]
            full = gather_full([self.checkpoint.tensors[n] for n in held])
            gathered = dict(zip(held, full))
            ready = None
            if full and full[0].is_cuda:
                ready = torch.cuda.Event()
                ready.record()
            for name in layer:
                tensor = gathered.pop(name, None)
                # the layer was gathered on the stream current at its first
                # next(); a consumer may resume us on another (the packed nccl
                # sender switches streams per chunk)
                if tensor is not None and ready is not None:
                    stream = torch.cuda.current_stream()
                    stream.wait_event(ready)
                    tensor.record_stream(stream)
                yield name, tensor


class ShardedM2NSource(M2NWeightSource):
    """Local shards and their placements, for nccl_m2n (no gather).

    m2n meshes have two axes, so the stage mesh's size-1 dimensions are dropped;
    ``engines.unsupported_reasons`` has already refused layouts that need more.
    """

    def __init__(
        self,
        checkpoint: ShardedCheckpoint,
        layout: TrainerLayout,
        rules: PlacementRules,
    ) -> None:
        self.checkpoint = checkpoint
        sizes = (layout.dp, layout.fsdp, layout.tp)
        axes = [i for i, size in enumerate(sizes) if size > 1]
        if len(axes) > 2:
            raise ValueError(f"nccl_m2n cannot describe layout {layout}")
        dims = [sizes[i] for i in axes] + [1] * (2 - len(axes))
        self._mesh = M2NMesh((dims[0], dims[1]), 0)
        self._metadata = []
        for t in checkpoint.manifest.tensors:
            spec = rules.placement_spec(t.name, len(t.shape))
            codes = [REPLICATE if spec[i] == "R" else int(spec[i][1:]) for i in axes]
            codes += [REPLICATE] * (2 - len(codes))
            placements = None if codes == [REPLICATE, REPLICATE] else tuple(codes)
            self._metadata.append(
                M2NParamMeta(
                    t.name,
                    _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
                    tuple(t.shape),
                    M2NLayout(self._mesh, placements),
                )
            )

    def mesh(self) -> M2NMesh:
        return self._mesh

    def metadata(self) -> list[ParamMeta]:
        return self._metadata

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for meta in self._metadata:
            yield meta.name, self.checkpoint.tensors[meta.name].to_local()


def _vllm_placements(spec: tuple[str, ...]) -> tuple:
    return tuple(VReplicate() if s == "R" else VShard(int(s[1:])) for s in spec)


def declared_metadata(checkpoint: ShardedCheckpoint) -> list[ParamMeta]:
    """Every tensor's owners and layout, identical on every rank: the ranks of
    its stage mesh (or its ep group's expert mesh), in mesh order, with the
    placements it is sharded by. Ordered by ``layerwise_groups``, so each
    gather group is contiguous."""
    layout, rules = checkpoint.layout, checkpoint.rules
    names = [t.name for t in checkpoint.manifest.tensors]
    stage_of = pp_stages(names, layout.pp)
    group_of = (
        ep_groups(names, layout.ep) if checkpoint.expert_mesh is not None else {}
    )
    grid = checkpoint.mesh.mesh
    metadata = {}
    for t in checkpoint.manifest.tensors:
        spec = rules.placement_spec(t.name, len(t.shape))
        stage = stage_of[t.name]
        if t.name in group_of:
            assert checkpoint.expert_mesh is not None
            ranks = checkpoint.expert_mesh.mesh[stage, group_of[t.name]]
            spec = spec[1:]
        else:
            ranks = grid[stage]
        metadata[t.name] = ParamMeta(
            t.name,
            _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
            tuple(t.shape),
            owners=tuple(ranks.flatten().tolist()),
            mesh_shape=tuple(ranks.shape),
            placements=_vllm_placements(spec),
        )
    return [metadata[n] for g in layerwise_groups(list(metadata)) for n in g]


class DeclaredGatherSource(WeightSource):
    """Full tensors via ``iter_full``, one batch per layer, with ownership
    declared in metadata instead of ``held_names()``.

    A rank yields only the names it owns; each layer is gathered among its
    owners, as ``gather_full`` does for an FSDP2 module.
    """

    def __init__(self, checkpoint: ShardedCheckpoint, batched: bool = True) -> None:
        self.checkpoint = checkpoint
        self.batched = batched
        self._metadata = declared_metadata(checkpoint)

    def metadata(self) -> list[ParamMeta]:
        return self._metadata

    def iter_full(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        held = self.checkpoint.tensors
        names = [m.name for m in self._metadata if m.name in held]
        for _, layer in itertools.groupby(names, key=_stack_key):
            layer = list(layer)
            if self.batched:
                full = gather_full([held[n] for n in layer])
            else:
                full = [held[n].full_tensor() for n in layer]
            yield list(zip(layer, full))


class DeclaredM2NSource(WeightSource):
    """Local shards via ``iter_local``, with each tensor's mesh and placements
    on the base ``ParamMeta``. Size-1 mesh axes are dropped, as m2n meshes
    have at most two."""

    def __init__(self, checkpoint: ShardedCheckpoint) -> None:
        self.checkpoint = checkpoint
        layout = checkpoint.layout
        sizes = (layout.dp, layout.fsdp, layout.tp)
        axes = [i for i, size in enumerate(sizes) if size > 1] or [0]
        owners = tuple(range(layout.world_size))
        self._metadata = []
        for t in checkpoint.manifest.tensors:
            spec = checkpoint.rules.placement_spec(t.name, len(t.shape))
            self._metadata.append(
                ParamMeta(
                    t.name,
                    _SAFETENSORS_DTYPE_TO_TORCH[t.safetensors_dtype][0],
                    tuple(t.shape),
                    owners=owners,
                    mesh_shape=tuple(sizes[i] for i in axes),
                    placements=_vllm_placements(tuple(spec[i] for i in axes)),
                )
            )

    def metadata(self) -> list[ParamMeta]:
        return self._metadata

    def iter_local(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        tensors = self.checkpoint.tensors
        names = [m.name for m in self._metadata]
        for _, layer in itertools.groupby(names, key=_stack_key):
            yield [(n, tensors[n].to_local()) for n in layer]
