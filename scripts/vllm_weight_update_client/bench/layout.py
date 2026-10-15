"""Trainer-side sharding layout: which slice of which tensor each rank holds.

The benchmark trainer has no model code. It holds checkpoint tensors the way a
sharded trainer would, as DTensors on a ``[dp, fsdp, tp]`` mesh per pipeline
stage:

- dp: ``Replicate`` (HSDP replica groups);
- fsdp: ``Shard(0)`` on every tensor with at least one dimension, as FSDP2 does;
- tp: ``Shard(0)`` for column-parallel tensors, ``Shard(1)`` for row-parallel
  ones, ``Replicate`` for the rest, chosen by name rules;
- pp: decoder layers split into contiguous stages; tensors outside the layers go
  to the first stage (embeddings) or the last (final norm, ``lm_head``);
- ep: the stage's ``dp * fsdp`` ranks form ``ep`` groups, each holding a
  contiguous block of experts on an ``[edp, tp]`` mesh, ``edp`` sharding like
  fsdp within the group.

Rank order is ``pp`` outermost, then ``dp``, ``fsdp`` and ``tp`` innermost, as
``init_device_mesh`` lays out ``(pp, dp, fsdp, tp)``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TpKind = Literal["col", "row", "rep"]
MESH_DIMS = ("pp", "dp", "fsdp", "tp")
EXPERT_MESH_DIMS = ("pp", "ep", "edp", "tp")

DEFAULT_TP_RULES: tuple[tuple[str, TpKind], ...] = (
    (
        r"\.(q_proj|k_proj|v_proj|qkv_proj|query_key_value|gate_proj|up_proj"
        r"|gate_up_proj|c_attn|c_fc|fc1|w1|w3)\.(weight|bias)$",
        "col",
    ),
    (r"(^|\.)(embed_tokens|wte|word_embeddings|lm_head)\.weight$", "col"),
    (r"\.(o_proj|down_proj|dense|c_proj|fc2|w2|out_proj)\.weight$", "row"),
)
_LAYER_INDEX = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")
_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")
_FIRST_STAGE = re.compile(r"(^|\.)(embed_tokens|wte|wpe|word_embeddings)\.")


@dataclass(frozen=True)
class TrainerLayout:
    pp: int = 1
    dp: int = 1
    fsdp: int = 1
    tp: int = 1
    ep: int = 1

    def __post_init__(self) -> None:
        for dim, size in asdict(self).items():
            if size < 1:
                raise ValueError(f"layout {dim} must be >= 1, got {size}")
        if (self.dp * self.fsdp) % self.ep:
            raise ValueError(f"ep={self.ep} must divide dp*fsdp={self.dp * self.fsdp}")

    @classmethod
    def parse(cls, spec: str) -> TrainerLayout:
        """``"fsdp=2,tp=2"``; unnamed dimensions are 1, ``""`` is one rank."""
        sizes: dict[str, int] = {}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            key, sep, value = part.partition("=")
            if not sep or key not in cls.__dataclass_fields__:
                raise ValueError(f"bad layout entry {part!r} in {spec!r}")
            sizes[key] = int(value)
        return cls(**sizes)

    def __str__(self) -> str:
        parts = [f"{dim}={size}" for dim, size in asdict(self).items() if size > 1]
        return ",".join(parts) or "single"

    @property
    def world_size(self) -> int:
        return self.pp * self.dp * self.fsdp * self.tp

    @property
    def mesh_shape(self) -> tuple[int, int, int, int]:
        return (self.pp, self.dp, self.fsdp, self.tp)

    @property
    def expert_mesh_shape(self) -> tuple[int, int, int, int]:
        return (self.pp, self.ep, self.dp * self.fsdp // self.ep, self.tp)

    @property
    def stage_mesh_dims(self) -> int:
        """Mesh dimensions within a stage that actually split ranks."""
        return sum(size > 1 for size in (self.dp, self.fsdp, self.tp))

    def pp_rank(self, rank: int) -> int:
        return rank // (self.dp * self.fsdp * self.tp)


@dataclass(frozen=True)
class PlacementRules:
    """Name rules for the tp placement; earlier rules win."""

    tp_rules: tuple[tuple[str, TpKind], ...] = DEFAULT_TP_RULES

    @classmethod
    def load(cls, path: Path | None) -> PlacementRules:
        """Rules from a JSON list of ``[regex, "col"|"row"|"rep"]``, checked
        before the defaults."""
        if path is None:
            return cls()
        extra = [(pattern, kind) for pattern, kind in json.loads(path.read_text())]
        for pattern, kind in extra:
            if kind not in ("col", "row", "rep"):
                raise ValueError(f"bad tp kind {kind!r} for {pattern!r}")
            re.compile(pattern)
        return cls(tuple(extra) + DEFAULT_TP_RULES)

    def tp_kind(self, name: str, ndim: int) -> TpKind:
        for pattern, kind in self.tp_rules:
            if re.search(pattern, name):
                if kind == "row" and ndim < 2:
                    return "rep"
                return kind if ndim >= 1 else "rep"
        return "rep"

    def placement_spec(self, name: str, ndim: int) -> tuple[str, str, str]:
        """Per stage-mesh dimension ``(dp, fsdp, tp)``: ``"R"`` or ``"S<dim>"``."""
        tp = {"col": "S0", "row": "S1", "rep": "R"}[self.tp_kind(name, ndim)]
        return ("R", "S0" if ndim >= 1 else "R", tp)


def pp_stages(names: Sequence[str], pp: int) -> dict[str, int]:
    """Pipeline stage of each tensor: contiguous decoder-layer ranges, with
    embeddings on the first stage and everything else outside the layers on
    the last."""
    layer_of = {}
    for name in names:
        match = _LAYER_INDEX.search(name)
        if match:
            layer_of[name] = int(match.group(1))
    num_layers = max(layer_of.values(), default=-1) + 1
    if pp > 1 and num_layers < pp:
        raise ValueError(f"pp={pp} needs at least {pp} layers, found {num_layers}")
    stages = {}
    for name in names:
        if name in layer_of:
            stages[name] = layer_of[name] * pp // num_layers
        elif _FIRST_STAGE.search(name):
            stages[name] = 0
        else:
            stages[name] = pp - 1
    return stages


def ep_groups(names: Sequence[str], ep: int) -> dict[str, int]:
    """Expert-parallel group of each routed-expert tensor: experts split into
    ``ep`` contiguous blocks. Tensors that are not experts are absent."""
    expert_of = {}
    for name in names:
        match = _EXPERT_INDEX.search(name)
        if match:
            expert_of[name] = int(match.group(1))
    num_experts = max(expert_of.values(), default=-1) + 1
    if ep > 1 and num_experts < ep:
        raise ValueError(f"ep={ep} needs at least {ep} experts, found {num_experts}")
    return {name: e * ep // num_experts for name, e in expert_of.items()}
