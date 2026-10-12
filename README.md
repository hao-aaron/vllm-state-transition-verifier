# vllm-state-transition-verifier

Standalone tooling for vLLM RL day-0 support:

1. a **trainer-free weight-update client** that pushes a Hugging Face checkpoint
   into a running vLLM server through vLLM's native NCCL weight-transfer engine;
2. an **RL experiment source-snapshot tool** that pins exactly which code a
   container ran.

Both were extracted from a private agent-infra repo with their original git
history intact. See [Placeholders](#placeholders) before using any default value.

## 1. Weight update client — `scripts/vllm_weight_update_client/`

Reads a local safetensors checkpoint in checkpoint-name order, builds a canonical
tensor manifest, stages one tensor at a time on the publisher GPU, and delegates
the transfer transaction to vLLM's trainer-side engine. No trainer, Ray, VIME,
Megatron, or Transformers dependency — it talks to vLLM directly.

| File | Purpose |
|---|---|
| `hf_checkpoint_nccl_publisher.py` | Checkpoint manifest, canonical source, stateful NCCL publisher |
| `run_vllm_weight_update.py` | Minimal CLI: one checkpoint NCCL weight-update transaction |
| `run_vllm_lifecycle.sh` | Optional wrapper orchestrating pause / reset / sleep / wake / resume around the publisher |
| `validation/` | Fixed-token oracle, GSM8K eval, pre/post comparison, and the refit-equals-fresh tensor audit |
| `provenance/write_run_sidecars.py` | Emits `.meta.json` provenance sidecars for a result directory |

Full usage notes are in
[`scripts/vllm_weight_update_client/README_ZH.md`](scripts/vllm_weight_update_client/README_ZH.md) (Chinese).

### Server contract

The server must use the NCCL backend with `packed` enabled; packed buffer size
and count use vLLM defaults, and the inference worker world size does not change
during an update:

```bash
--weight-transfer-config '{"backend":"nccl","packed":true}'
```

Client and server containers must share the same vLLM runtime.
The publisher trainer rank and serving ranks must use different physical GPUs.

### Minimal run

The checkpoint must already sit on the publisher node's local NVMe and be passed
explicitly — there is no implicit download, and reading from a network filesystem
defeats the direct-file host-to-device path.

```bash
python scripts/vllm_weight_update_client/run_vllm_weight_update.py \
  --base-url http://<server-host>:8000 \
  --model <local-model-path> \
  --revision <revision> \
  --checkpoint-path <local-safetensors-checkpoint> \
  --device cuda:0 \
  --output <result-dir>/weight-update.json
```

The publisher only runs `start_weight_update → metadata/update buckets →
finish_weight_update`. It does not pause, sleep, clear caches, send requests, or
resume on your behalf, and it never calls `finish` on a failed transfer — a
partial update is never reported as success.

### Refit-equals-fresh tensor audit

The fixed-token oracle sees only what one short completion depends on. The
tensor audit checks every device tensor instead: after an NCCL update, the
server must hold the same bytes as a server started fresh on the published
checkpoint.

| File (`validation/`) | Purpose |
|---|---|
| `tensor_audit_extension.py` | vLLM worker extension: pointer, layout and on-device digest of every parameter, buffer and kernel-held (off-registry) tensor, plus allocator stats |
| `tensor_audit.py` | `snapshot` over `/collective_rpc`; `compare` one update step into a report |
| `make_perturbed_checkpoint.py` | Variant of a checkpoint in which every tensor changes, scales and norms included |
| `run_refit_audit.py` | Single-node driver: A → B → C over NCCL, fresh servers on B and C, report |

Each update step must satisfy:

- **equal to fresh**: every registered and off-registry tensor has the same
  bytes and layout as on the fresh server (kernel-held derived state such as
  MoE alphas or `1/a_scale` included);
- **stable**: no tensor that existed before the update moved (CUDA graphs keep
  reading old addresses), appeared or disappeared;
- **no leak**: allocated device memory grew by at most 64 MiB (configurable).

Tensors whose bytes the update did not change are listed as **untested**: a
stale copy of them would pass. Tensors that change between two snapshots of
the same server with only requests in between are runtime state (workspaces)
and are excluded and listed.

```bash
V=scripts/vllm_weight_update_client/validation
python $V/make_perturbed_checkpoint.py <A-dir> <B-dir> --variant 1
python $V/make_perturbed_checkpoint.py <A-dir> <C-dir> --variant 2
python $V/run_refit_audit.py --model-a <A-dir> --update <B-dir> --update <C-dir> \
  --server-gpus 0 --publisher-gpu 1 --output-dir <result-dir> \
  -- --max-model-len 4096 --gpu-memory-utilization 0.7
```

`<result-dir>/report.md` summarizes the run for a PR; `report.json` lists every
tensor by check. The server's `--gpu-memory-utilization` must leave room for
the digest temporaries (about 0.5 GiB). The driver starts the servers itself
(`VLLM_SERVER_DEV_MODE=1`, `--worker-extension-cls
tensor_audit_extension.TensorAuditExtension`); for other topologies, run
`tensor_audit.py snapshot` / `compare` against servers you start.

## 2. Source snapshot — `scripts/rl_source_snapshot.py`

Before an experiment starts, turns the current (possibly uncommitted) state of a
change capsule into a snapshot commit and a clean worktree, without disturbing
the original worktree. The container then mounts only that worktree, and the tool
verifies the real import path and mounted file hashes match.

```bash
python scripts/rl_source_snapshot.py {prepare,verify,render-check,image-digest,cleanup} --help
```

A branch name is not sufficient provenance: results should record branch,
snapshot commit, base SHA, patch, tree hash, and image digest.

## Placeholders

Internal model codenames, cluster paths, registry namespaces, and host addresses
were replaced with placeholders throughout the entire history:

| Placeholder | Was |
|---|---|
| `ModelA` / `modela`, `ModelB` / `modelb`, `DraftModel` | Internal model codenames |
| `<registry>` | A private container registry namespace |
| `/mnt/<shared-fs>`, `/path/to/vllm` | Cluster filesystem paths |
| `127.0.0.1` | A cluster host address |

**These are placeholders, not working defaults.** Anything that looks like a
default image, path, or host must be set explicitly for your environment.

## Tests

```bash
uv venv --python 3.12
uv pip install pyyaml pytest
.venv/bin/python -m pytest tests/ -v
```

`hf_checkpoint_nccl_publisher.py` additionally needs `torch`, `safetensors`, and
a vLLM install providing `vllm.distributed.weight_transfer`; it is exercised
against a live server, not in this test suite. The tensor-audit tests run on
CPU; their digest and perturbation cases need `torch` and `safetensors` and are
skipped without them.

## Scope

This is the first batch — scripts only. The day-0 RL model-support checklist and
the staged weight-update validation protocol (L0–L6 lanes, PASS / FAIL_CLOSED
criteria) are planned as a follow-up.
