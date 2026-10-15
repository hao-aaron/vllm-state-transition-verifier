from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path[:0] = [
    str(Path(__file__).parents[1] / "scripts" / "vllm_weight_update_client" / p)
    for p in ("bench", "")
]

from engines import unsupported_reasons  # noqa: E402
from layout import PlacementRules, TrainerLayout, pp_stages  # noqa: E402


def test_layout_parse_and_world_size():
    layout = TrainerLayout.parse("pp=2,fsdp=2,tp=2")
    assert layout.world_size == 8
    assert layout.mesh_shape == (2, 1, 2, 2)
    assert [layout.pp_rank(r) for r in (0, 3, 4, 7)] == [0, 0, 1, 1]
    assert str(layout) == "pp=2,fsdp=2,tp=2"
    assert TrainerLayout.parse("").world_size == 1
    with pytest.raises(ValueError):
        TrainerLayout.parse("zp=2")
    with pytest.raises(ValueError):
        TrainerLayout.parse("fsdp=2,ep=4")


@pytest.mark.parametrize(
    "name, ndim, kind",
    [
        ("model.layers.0.self_attn.q_proj.weight", 2, "col"),
        ("model.layers.0.self_attn.k_proj.bias", 1, "col"),
        ("model.layers.0.self_attn.o_proj.weight", 2, "row"),
        ("model.layers.0.mlp.experts.3.down_proj.weight", 2, "row"),
        ("model.layers.0.mlp.gate_up_proj.weight", 2, "col"),
        ("model.embed_tokens.weight", 2, "col"),
        ("lm_head.weight", 2, "col"),
        ("model.layers.0.input_layernorm.weight", 1, "rep"),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 2, "rep"),
        ("model.layers.0.self_attn.q_norm.weight", 1, "rep"),
    ],
)
def test_default_tp_rules(name, ndim, kind):
    assert PlacementRules().tp_kind(name, ndim) == kind


def test_placement_rule_overrides(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('[["kv_a_proj_with_mqa", "col"]]')
    rules = PlacementRules.load(path)
    name = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    assert rules.placement_spec(name, 2) == ("R", "S0", "S0")
    assert rules.placement_spec("model.norm.weight", 1) == ("R", "S0", "R")
    assert rules.placement_spec("scalar", 0) == ("R", "R", "R")


def test_pp_stages():
    names = [
        "model.embed_tokens.weight",
        *(f"model.layers.{i}.mlp.up_proj.weight" for i in range(4)),
        "model.norm.weight",
        "lm_head.weight",
    ]
    stages = pp_stages(names, 2)
    assert [stages[n] for n in names] == [0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        pp_stages(names, 5)


def test_engines_report_what_they_cannot_serve():
    assert unsupported_reasons(TrainerLayout.parse("fsdp=2,tp=2"), "nccl") == []
    assert unsupported_reasons(TrainerLayout(), "nccl", "ray") == []
    (reason,) = unsupported_reasons(TrainerLayout.parse("pp=2"), "nccl")
    assert "multi_sender" in reason and "partial_ownership" in reason
    assert any(
        "--launcher ray" in r
        for r in unsupported_reasons(TrainerLayout(), "sharded_rdt", "torchrun")
    )
    assert not any(
        "launcher" in r
        for r in unsupported_reasons(TrainerLayout(), "sharded_rdt", "ray")
    )
    assert any(
        "mesh dims" in r
        for r in unsupported_reasons(TrainerLayout(dp=2, fsdp=2, tp=2), "nccl_m2n")
    )
    m2n_2d = unsupported_reasons(TrainerLayout(fsdp=2, tp=2), "nccl_m2n")
    assert any("fsdp and tp both shard" in r for r in m2n_2d)
    assert unsupported_reasons(TrainerLayout(dp=2, fsdp=2), "nccl_m2n") == []


def _sharded_worker(rank: int, world: int, port: int, ckpt: str, spec: str, q):
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from hf_checkpoint_nccl_publisher import CheckpointManifest
        from sharded_checkpoint import GatherSource, ShardedCheckpoint, build_meshes

        layout = TrainerLayout.parse(spec)
        mesh, expert_mesh = build_meshes(layout, "cpu")
        manifest = CheckpointManifest.load(
            model=ckpt, revision="local", checkpoint_path=ckpt
        )
        checkpoint = ShardedCheckpoint(
            manifest, layout, mesh, PlacementRules(), torch.device("cpu"), expert_mesh
        )
        source = GatherSource(checkpoint, partial=layout.pp > 1 or layout.ep > 1)
        # plain lists: tensors in the queue would die with this process
        out = {name: None if t is None else t.tolist() for name, t in source}
        q.put((rank, out, checkpoint.local_bytes, source.held_names()))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "spec",
    ["fsdp=2,tp=2", "dp=2,tp=2", "pp=2,fsdp=2", "fsdp=2,ep=2,tp=2", "pp=2,dp=2,ep=2"],
)
def test_sharded_checkpoint_gathers_back_to_the_checkpoint(tmp_path, spec):
    """Every rank reads only its slice, and full_tensor() reassembles exactly
    the checkpoint tensor, uneven shapes included."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from safetensors.torch import save_file

    tensors = {
        "model.embed_tokens.weight": torch.randn(7, 4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(5, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 5),
        "model.layers.0.input_layernorm.weight": torch.randn(3),
        "model.layers.1.mlp.down_proj.weight": torch.randn(4, 3),
        **{
            f"model.layers.1.mlp.experts.{e}.{proj}.weight": torch.randn(*shape)
            for e in range(4)
            for proj, shape in (("gate_proj", (3, 4)), ("down_proj", (4, 3)))
        },
        "model.norm.weight": torch.randn(4),
        "lm_head.weight": torch.randn(7, 4),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    world = TrainerLayout.parse(spec).world_size
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    procs = [
        ctx.Process(
            target=_sharded_worker, args=(r, world, port, str(tmp_path), spec, q)
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    layout = TrainerLayout.parse(spec)
    copies = {  # ranks holding a full copy of each byte: replicated dims only
        # experts live on one ep group, sharded over it (edp) and tp
        name: (1 if layout.ep > 1 and ".experts." in name else layout.dp)
        * (layout.tp if PlacementRules().tp_kind(name, t.ndim) == "rep" else 1)
        for name, t in tensors.items()
    }
    expected = sum(t.numel() * 4 * copies[n] for n, t in tensors.items())
    assert sum(local_bytes for _, _, local_bytes, _ in results) == expected
    for rank, out, _, held in results:
        assert list(out) == sorted(out, key=list(out).index)  # manifest order
        for name, full in out.items():
            if full is None:
                assert name not in held
            else:
                assert full == tensors[name].tolist(), (rank, name)
        if layout.pp > 1:
            stage = layout.pp_rank(rank)
            assert ("model.embed_tokens.weight" in held) == (stage == 0)
            assert ("lm_head.weight" in held) == (stage == 1)
        if layout.ep > 1:
            stage_rank = rank % (layout.dp * layout.fsdp * layout.tp)
            group = stage_rank // layout.tp // (layout.dp * layout.fsdp // layout.ep)
            for e in range(4):
                name = f"model.layers.1.mlp.experts.{e}.gate_proj.weight"
                in_stage = layout.pp_rank(rank) == layout.pp - 1 or layout.pp == 1
                assert (name in held) == (in_stage and e * layout.ep // 4 == group)
