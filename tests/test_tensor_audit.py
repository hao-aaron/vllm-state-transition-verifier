from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "vllm_weight_update_client"
        / "validation"
    ),
)

import tensor_audit  # noqa: E402


def _entry(digest: str, ptr: int = 1, shape=(4,)) -> dict:
    return {
        "ptr": ptr,
        "shape": list(shape),
        "stride": [1],
        "dtype": "bfloat16",
        "digest": digest,
    }


def _snap(
    registered: dict, off_registry: dict | None = None, allocated: int = 0
) -> list:
    return [
        {
            "rank": 0,
            "registered": registered,
            "off_registry": off_registry or {},
            "memory": {"allocated": allocated},
        }
    ]


def test_update_that_matches_fresh_passes():
    before = _snap(
        {"w": _entry("a"), "s": _entry("a_s")}, {"k.alpha": _entry("a_al", 9)}
    )
    after = _snap(
        {"w": _entry("b"), "s": _entry("b_s")}, {"k.alpha": _entry("b_al", 9)}
    )
    fresh = _snap(
        {"w": _entry("b", 5), "s": _entry("b_s", 6)}, {"k.alpha": _entry("b_al", 7)}
    )
    step = tensor_audit.compare_step(before, after, fresh)
    assert step["status"] == "PASS"
    assert step["tensors_exercised"] == step["tensors_compared"] == 3


def test_stale_derived_state_fails():
    """A kernel-held derived value keeps A's bytes after the update."""
    before = _snap({"s": _entry("a_s")}, {"k.alpha": _entry("a_al")})
    after = _snap({"s": _entry("b_s")}, {"k.alpha": _entry("a_al")})
    fresh = _snap({"s": _entry("b_s")}, {"k.alpha": _entry("b_al")})
    step = tensor_audit.compare_step(before, after, fresh)
    assert step["status"] == "FAIL"
    assert step["findings"]["mismatched"]["off_registry"] == ["k.alpha"]


def test_moved_tensor_and_memory_growth_fail():
    before = _snap({"w": _entry("a", ptr=1)}, allocated=100)
    after = _snap({"w": _entry("b", ptr=2)}, allocated=100 + (128 << 20))
    fresh = _snap({"w": _entry("b", ptr=3)})
    step = tensor_audit.compare_step(before, after, fresh)
    assert set(step["failing_checks"]) == {"moved", "memory"}


def test_unchanged_tensor_is_untested_not_failed():
    before = _snap({"w": _entry("a"), "norm": _entry("n")})
    after = _snap({"w": _entry("b"), "norm": _entry("n")})
    fresh = _snap({"w": _entry("b"), "norm": _entry("n")})
    step = tensor_audit.compare_step(before, after, fresh)
    assert step["status"] == "PASS"
    assert step["findings"]["untested"]["registered"] == ["norm"]
    assert step["tensors_exercised"] == 1


def test_runtime_state_is_excluded():
    a1 = _snap({"w": _entry("a")}, {"ws": _entry("x")})
    a2 = _snap({"w": _entry("a")}, {"ws": _entry("y")})
    runtime = tensor_audit.runtime_names(a1, a2)
    assert runtime == {"registered": set(), "off_registry": {"ws"}}
    after = _snap({"w": _entry("b")}, {"ws": _entry("z")})
    fresh = _snap({"w": _entry("b")}, {"ws": _entry("q")})
    assert tensor_audit.compare_step(a2, after, fresh, runtime)["status"] == "PASS"


def test_structure_change_fails():
    before = _snap({"w": _entry("a"), "raw": _entry("r")})
    after = _snap({"w": _entry("b")})
    fresh = _snap({"w": _entry("b"), "raw": _entry("r2")})
    step = tensor_audit.compare_step(before, after, fresh)
    assert {"disappeared", "missing_vs_fresh"} <= set(step["failing_checks"])


def test_markdown_lists_failures():
    before = _snap({"w": _entry("a")})
    step = tensor_audit.compare_step(
        before, _snap({"w": _entry("a")}), _snap({"w": _entry("b")})
    )
    step["name"] = "update1"
    md = tensor_audit.render_markdown({"status": "FAIL", "steps": [step]})
    assert "update1: mismatched (1)" in md and "`w`" in md


def test_device_digest_sees_every_word_and_ignores_reduction_order():
    torch = pytest.importorskip("torch")
    from tensor_audit_extension import _device_digest

    data = torch.arange(1 << 12, dtype=torch.int32).view(torch.uint8)
    base = _device_digest(data.clone())
    assert _device_digest(data.clone()) == base
    for index in (0, 4097, data.numel() - 1):
        changed = data.clone()
        changed[index] ^= 1
        assert _device_digest(changed) != base
    swapped = data.view(torch.int32).clone()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert _device_digest(swapped.view(torch.uint8)) != base


def test_perturbation_changes_every_payload_and_keeps_metadata():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from make_perturbed_checkpoint import perturb_tensor

    cases = {
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "layer.bias": torch.zeros(8, dtype=torch.float32),
        "layer.weight_scale": torch.full((4,), 0.02, dtype=torch.float32),
        "layer.qweight": torch.arange(8, dtype=torch.int32),
        "experts.blocks": torch.arange(256, dtype=torch.int32).to(torch.uint8),
    }
    if hasattr(torch, "float8_e4m3fn"):
        cases["layer.weight"] = torch.linspace(-4, 4, 16).to(torch.float8_e4m3fn)
    for name, t in cases.items():
        b, how_b = perturb_tensor(name, t, 1, 1 / 32)
        c, _ = perturb_tensor(name, t, 2, 1 / 32)
        raw = lambda x: x.contiguous().view(torch.uint8)  # noqa: E731
        assert not how_b.startswith("unchanged"), name
        assert b.shape == t.shape and b.dtype == t.dtype, name
        assert not torch.equal(raw(b), raw(t)), name
        assert not torch.equal(raw(b), raw(c)), name
    blocks, _ = perturb_tensor(
        "experts.scales", torch.full((4,), 0xFE, dtype=torch.uint8), 1, 0.03
    )
    assert not torch.any(blocks == 0xFF)  # e8m0 NaN
    for name, t in {
        "layer.g_idx": torch.arange(8, dtype=torch.int32),
        "layer.weight_shape": torch.tensor([4, 8]),
    }.items():
        out, how = perturb_tensor(name, t, 1, 1 / 32)
        assert how.startswith("unchanged") and torch.equal(out, t), name
