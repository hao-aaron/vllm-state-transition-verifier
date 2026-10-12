"""vLLM worker extension: snapshot every device tensor the model can read.

Load it into the server under test:

    PYTHONPATH=<repo>/scripts/vllm_weight_update_client/validation \
    VLLM_SERVER_DEV_MODE=1 vllm serve ... \
        --worker-extension-cls tensor_audit_extension.TensorAuditExtension

and call it through the dev endpoint (one result per worker rank):

    POST /collective_rpc {"method": "audit_snapshot"}

A snapshot covers two kinds of tensors:

- registered: every parameter and buffer of the model;
- off_registry: device tensors reachable from module attributes that are not
  registered (quant methods, kernels, quant configs, attention impls, ...).
  Kernels keep weight-derived values here (MoE alphas, ``1/a_scale``, FlashInfer
  bmm scales), and a weight update that misses them leaves stale values that
  CUDA graphs keep reading.

Each tensor records its data_ptr, shape, stride, dtype and a content digest
computed on the device, plus per-rank allocator statistics. The client
(``tensor_audit.py``) diffs snapshots taken across an update and against a
fresh server.
"""

from __future__ import annotations

import gc
import hashlib
from typing import Any

import torch

# Tensors at most this large are hashed with md5 on the host; larger ones with
# a position-weighted sum on the device (see _device_digest).
_HOST_DIGEST_MAX_BYTES = 1 << 20
_CHUNK_WORDS = 1 << 24
_MASK64 = (1 << 64) - 1
# Odd multipliers make every single-word change visible: an odd coefficient is
# invertible mod 2**64, so coefficient * delta != 0 for any nonzero delta.
_MULTIPLIERS = (
    (0x9E3779B97F4A7C15, 0x632BE59BD9B4E019),
    (0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9),
)
# Attribute paths holding runtime state rather than weights.
_RUNTIME_PATH_MARKERS = ("kv_cache", "kv_caches")
_MAX_DEPTH = 5
_MAX_DICT_ITEMS = 64


def _signed64(value: int) -> int:
    value &= _MASK64
    return value - (1 << 64) if value >= 1 << 63 else value


def _device_digest(flat_bytes: torch.Tensor) -> str:
    pad = (-flat_bytes.numel()) % 4
    if pad:
        flat_bytes = torch.cat([flat_bytes, flat_bytes.new_zeros(pad)])
    words = flat_bytes.view(torch.int32)
    sums = [0] * len(_MULTIPLIERS)
    for start in range(0, words.numel(), _CHUNK_WORDS):
        chunk = words[start : start + _CHUNK_WORDS].to(torch.int64)
        index = torch.arange(
            start, start + chunk.numel(), device=chunk.device, dtype=torch.int64
        )
        for i, (mul, add) in enumerate(_MULTIPLIERS):
            coeff = (index * _signed64(mul) + _signed64(add)) | 1
            # int64 products and sums wrap mod 2**64, so the result does not
            # depend on the reduction order.
            sums[i] += int((chunk * coeff).sum().item())
        del chunk, index
    return "dev:" + "".join(f"{s & _MASK64:016x}" for s in sums)


def tensor_digest(t: torch.Tensor) -> str:
    """Content digest of `t`'s bytes; equal digests mean equal bytes (with
    overwhelming probability), any single changed 4-byte word always shows."""
    t = t.detach()
    if t.is_meta:
        return f"meta{tuple(t.shape)}"
    if t.numel() == 0:
        return f"empty{tuple(t.shape)}"
    flat = t.contiguous().view(-1).view(torch.uint8)
    if flat.numel() <= _HOST_DIGEST_MAX_BYTES or flat.device.type != "cuda":
        return "md5:" + hashlib.md5(flat.cpu().numpy().tobytes()).hexdigest()
    return _device_digest(flat)


def _describe(t: torch.Tensor, digest: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ptr": t.data_ptr() if not t.is_meta else 0,
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
        "device": str(t.device),
    }
    if digest:
        entry["digest"] = tensor_digest(t)
    return entry


def collect_off_registry(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Device tensors reachable from module attributes that are not registered
    parameters or buffers, keyed by attribute path."""
    registered = {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}
    out: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(obj: Any, path: str, depth: int) -> None:
        if depth > _MAX_DEPTH or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if (
                id(obj) not in registered
                and obj.device.type != "cpu"
                and not any(m in path for m in _RUNTIME_PATH_MARKERS)
            ):
                out[path] = obj
            return
        if isinstance(obj, torch.nn.Module):
            return  # reached from the module walk below
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:_MAX_DICT_ITEMS]:
                visit(v, f"{path}[{k!r}]", depth + 1)
            return
        attrs = getattr(obj, "__dict__", None)
        if attrs is None or type(obj).__module__.startswith(("torch", "builtins")):
            return
        for k, v in list(attrs.items()):
            visit(v, f"{path}.{k}", depth + 1)

    for module_name, module in model.named_modules():
        for k, v in module.__dict__.items():
            if k in ("_parameters", "_buffers", "_modules"):
                continue
            visit(v, f"{module_name}.{k}" if module_name else k, 0)
    return out


class TensorAuditExtension:
    """Methods callable through ``collective_rpc`` (string arguments only)."""

    rank: int
    get_model: Any
    model_runner: Any

    def _audit_model(self, target: str) -> torch.nn.Module:
        if target == "draft":
            return self.model_runner.get_draft_model()
        if target != "target":
            raise ValueError(f"unknown audit target {target!r}")
        return self.get_model()

    @torch.no_grad()
    def audit_snapshot(self, digests: str = "1", target: str = "target") -> dict:
        """Pointers, layouts and (optionally) content digests of every
        registered and off-registry device tensor, plus allocator statistics."""
        with_digests = digests not in ("0", "false", "False", "")
        model = self._audit_model(target)
        gc.collect()
        memory = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # before the digests, whose temporaries would count
            free, total = torch.cuda.mem_get_info()
            memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "device_free": free,
                "device_total": total,
            }
        registered = {
            name: _describe(t, with_digests)
            for name, t in [*model.named_parameters(), *model.named_buffers()]
            if t.device.type != "cpu"
        }
        off_registry = {}
        for path, t in collect_off_registry(model).items():
            try:
                off_registry[path] = _describe(t, with_digests)
            except Exception as exc:  # noqa: BLE001 (exotic subclasses)
                off_registry[path] = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "rank": getattr(self, "rank", 0),
            "target": target,
            "registered": registered,
            "off_registry": off_registry,
            "memory": memory,
        }
