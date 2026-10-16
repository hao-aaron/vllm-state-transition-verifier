#!/usr/bin/env python3
"""Write a variant of a safetensors checkpoint in which every tensor changes.

A weight-update test is only as strong as its update: a tensor whose value is
the same before and after the update can be stale without anyone noticing.
Checkpoints that perturb only the large weights leave scales, norms, biases,
attention sinks and KV-cache scales unchanged, which hides exactly the
weight-derived state a reload most often gets wrong (``w_scale * a_scale``
alphas, ``1 / a_scale``, FlashInfer bmm scales). This tool changes every tensor
it can change without breaking the checkpoint's structure:

- floating point (bf16/fp16/fp32/fp64): scaled by ``1 + k * eps``; a tensor
  that is all zeros gets ``k * eps`` added; quantization scales (``*scale*``)
  are scaled by ``2 ** k`` instead, since servers may round them to powers of
  two (UE8M0 block scales), which absorbs a small relative change;
- float8 (e4m3/e5m2): scaled through float32 and clamped to the dtype's range;
- float8_e8m0 and 1-byte integer payloads (packed FP4/MXFP4 blocks, e8m0
  scales stored as uint8, int8 weights): low bits flipped (``xor k``), keeping
  0xFF (NaN in e8m0) out of the result;
- int32/int16 payloads (GPTQ/AWQ ``qweight``/``qzeros``): low bits flipped;
- int64 tensors, bool tensors and index-like names (``g_idx``, ``perm``,
  ``weight_shape``, ...): copied unchanged and listed as such.

``--variant k`` (1, 2, ...) picks the perturbation, so variants differ from the
source and from each other in every perturbed tensor. Identical source tensors
(tied embeddings) stay identical. Every file that is not a safetensors shard
(config, tokenizer, index) is copied as is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

UNCHANGED_NAME_MARKERS = (
    "g_idx",
    "perm",
    "weight_shape",
    "expert_map",
    "_idx",
    "indices",
)
FLOAT8_DTYPES = {
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, name)
}
E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _flip_bytes(t: torch.Tensor, k: int) -> torch.Tensor:
    raw = t.contiguous().view(torch.uint8)
    flipped = raw ^ k
    # 0xFF is NaN in e8m0 (MXFP4/MXFP8 scales are often stored as uint8)
    flipped = torch.where(flipped == 0xFF, raw ^ (k << 1), flipped)
    return flipped.view(t.dtype).view(t.shape)


def perturb_tensor(
    name: str, t: torch.Tensor, variant: int, eps: float
) -> tuple[torch.Tensor, str]:
    """Return the perturbed tensor and how it was changed ("unchanged: ..."
    when it is deliberately left alone)."""
    if variant < 1:
        raise ValueError("variant must be >= 1")
    leaf = name.rsplit(".", 1)[-1]
    if t.numel() == 0:
        return t, "unchanged: empty"
    if t.dtype in (torch.int64, torch.bool) or any(
        m in leaf for m in UNCHANGED_NAME_MARKERS
    ):
        return t, f"unchanged: {str(t.dtype).removeprefix('torch.')} metadata/index"
    if t.dtype in FLOAT8_DTYPES:
        info = torch.finfo(t.dtype)
        factor = 1.0 + variant * max(eps, 0.15)  # 3 mantissa bits: ~12% steps
        out = (t.float() * factor).clamp(info.min, info.max).to(t.dtype)
        if torch.equal(out.view(torch.uint8), t.view(torch.uint8)):
            out = _flip_bytes(t, variant)
        return out, f"float8 x{factor:g}"
    if t.dtype == E8M0 or t.dtype in (torch.uint8, torch.int8):
        return _flip_bytes(t, variant), f"bytes xor {variant}"
    if t.dtype in (torch.int32, torch.int16):
        return t ^ variant, f"int xor {variant}"
    if t.is_floating_point():
        if not torch.any(t != 0):
            return t + variant * eps, f"zeros + {variant * eps:g}"
        factor = 2.0**variant if "scale" in leaf else 1.0 + variant * eps
        out = (t.to(torch.float32) * factor).to(t.dtype)
        return out, f"x{factor:g}"
    return t, f"unchanged: unsupported dtype {t.dtype}"


def perturb_checkpoint(
    source: Path, dest: Path, variant: int, eps: float
) -> dict[str, str]:
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files in {source}")
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy2(path, dest / path.name)
    changes: dict[str, str] = {}
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for name in f.keys():
                out, how = perturb_tensor(name, f.get_tensor(name), variant, eps)
                tensors[name] = out.contiguous()
                changes[name] = how
        save_file(tensors, str(dest / shard.name), metadata=metadata)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="local checkpoint directory")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--variant", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1 / 32)
    args = parser.parse_args()
    changes = perturb_checkpoint(args.source, args.dest, args.variant, args.eps)
    unchanged = sorted(n for n, how in changes.items() if how.startswith("unchanged"))
    report = {
        "source": str(args.source),
        "variant": args.variant,
        "eps": args.eps,
        "num_tensors": len(changes),
        "unchanged": {n: changes[n] for n in unchanged},
        "changes": changes,
    }
    (args.dest / "perturbation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dest": str(args.dest),
                "tensors": len(changes),
                "unchanged": len(unchanged),
            }
        )
    )


if __name__ == "__main__":
    main()
