"""Hold a safetensors checkpoint as DTensors, reading only each rank's slice."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
from layout import PlacementRules, TrainerLayout, pp_stages
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from vllm.distributed.weight_transfer import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.base import layerwise_groups
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    M2NMesh,
    M2NParamMeta,
)
from vllm.distributed.weight_transfer.m2n_source import M2NWeightSource

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


class ShardedCheckpoint:
    """This rank's share of a checkpoint under ``layout``.

    Tensors of other pipeline stages are not held. ``load`` refills the local
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
    ) -> None:
        self.manifest = manifest
        stage = layout.pp_rank(dist.get_rank())
        stage_mesh = mesh["dp", "fsdp", "tp"]
        stage_of = pp_stages([t.name for t in manifest.tensors], layout.pp)
        self.tensors: dict[str, DTensor] = {}
        self._slices: dict[str, tuple[slice, ...]] = {}
        for meta in manifest.tensors:
            if stage_of[meta.name] != stage:
                continue
            placements = _placements(rules.placement_spec(meta.name, len(meta.shape)))
            local_shape, offset = compute_local_shape_and_global_offset(
                meta.shape, stage_mesh, placements
            )
            local = torch.empty(
                local_shape,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[meta.safetensors_dtype][0],
                device=device,
            )
            self.tensors[meta.name] = DTensor.from_local(
                local,
                stage_mesh,
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


class GatherSource(WeightSource):
    """Full tensors gathered with ``full_tensor()``.

    Without ``partial`` every rank holds every tensor and iterates in manifest
    order. With ``partial`` (engines with partial ownership, e.g. a pp stage
    holding its layers), metadata is the manifest in ``layerwise_groups``
    order, so each gather group is contiguous; a rank iterates only the groups
    it holds part of, yielding ``None`` for names held elsewhere.
    """

    def __init__(self, checkpoint: ShardedCheckpoint, partial: bool) -> None:
        self.checkpoint = checkpoint
        self.partial = partial
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
        for name in names:
            tensor = self.checkpoint.tensors.get(name)
            yield name, None if tensor is None else tensor.full_tensor()


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
                    placements,
                )
            )

    def mesh(self) -> M2NMesh:
        return self._mesh

    def metadata(self) -> list[ParamMeta]:
        return self._metadata

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for meta in self._metadata:
            yield meta.name, self.checkpoint.tensors[meta.name].to_local()
