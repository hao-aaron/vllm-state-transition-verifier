# Weight-transfer engine benchmark

Measures what one vLLM weight sync costs for a given engine and trainer layout,
and optionally checks that the synced server holds the same bytes as a server
started fresh on the same checkpoint.

The trainer is synthetic. There is no model code and no training: each rank
reads only its slice of a safetensors checkpoint into DTensors, laid out the way
a `pp × dp × fsdp × tp × ep` trainer would hold them. It then pushes the slices
through the engine's own trainer API (`WeightTransferTrainerFactory.trainer_init`
then `send_weights()`). The timing is therefore the engine's cost for that
layout, not the harness's.

## Quick start

```bash
B=scripts/vllm_weight_update_client/bench
V=scripts/vllm_weight_update_client/validation
PY=<venv with vLLM>/bin/python
export TMPDIR=/mnt/data/$USER/tmp RAY_TMPDIR=/mnt/data/$USER/ray

# 1. Two variants of the checkpoint, so every sync really changes every tensor.
$PY $V/make_perturbed_checkpoint.py <A> <A>-s1 --variant 1
$PY $V/make_perturbed_checkpoint.py <A> <A>-s2 --variant 2

# 2. Server TP=2 on GPUs 0,1, trainer fsdp=2 on GPUs 2,3, audit at the end.
$PY $B/run_bench.py --python $PY \
  --model-a <A> --update <A>-s1 --update <A>-s2 \
  --backend nccl --layout fsdp=2 \
  --server-gpus 0,1 --trainer-gpus 2,3 --warmup 1 --iters 3 --audit \
  --output-dir runs/nccl-fsdp2 \
  -- --tensor-parallel-size 2 --max-model-len 2048 --gpu-memory-utilization 0.5
```

Everything after `--` is passed to `vllm serve` unchanged. The driver adds
`--weight-transfer-config`, the audit worker extension and whatever the engine
needs, such as the Ray executor for `sharded_rdt`.

The exit code is 0 on success, or when there is no audit. It is 1 if the audit
did not PASS. An unsupported engine/layout combination exits before anything
launches, with each missing capability named.

## Requirements

- **One vLLM install** shared by the driver, the server and every trainer rank.
  `--python` chooses the interpreter for the server and the trainer (by
  default the driver's own). The harness imports
  `vllm.distributed.weight_transfer`, so it tracks vLLM main. Results record
  `vllm_version`.
- **GPUs:** the server and trainer use separate GPUs, except for `ipc`, which
  needs `--trainer-gpus` equal to `--server-gpus`. `--trainer-gpus` must list
  exactly `pp·dp·fsdp·tp` GPUs.
- **Checkpoints** must be on local disk (NVMe), not a network filesystem.
  `--model-a` is what the server starts on. Each `--update` is sent in turn:
  sync i sends `update[i % n]`. All checkpoints need the same tensor names and
  shapes.
- **Temp space:** nvcc, inductor and Python use `TMPDIR`, and Ray uses
  `RAY_TMPDIR` (or `--ray-temp-dir`). On hosts with a small root disk, point
  both at local scratch first, or runs fail with confusing errors.
- **Per engine:**

| Engine | Extra needs | Layouts accepted |
|---|---|---|
| `nccl` | — | dp, fsdp, tp (one sender: pp, ep refused) |
| `ipc` | colocated; the driver sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | dp, fsdp, tp |
| `nccl_m2n` | `nccl-extensions`, NCCL ≥ 2.30.5 in every process (see below) | dp with fsdp *or* tp; mesh ≤ 2-D, ≤ 16 source shards |
| `sharded_rdt` | Ray ≥ 2.56, `nixl`; `--launcher ray` | dp, fsdp, tp, pp, ep |
| `nixl_slot` | `nixl`; branch `nixl-slot-explore` plus a vLLM build that has `vllm/distributed/weight_transfer/nixl_slot.py` (not upstream) | dp, fsdp, tp, pp, ep |

  For `nccl_m2n`, preload the NCCL that `libnccl_m2n.so` was built against, e.g.
  `libnccl.so.2` unpacked from the `nvidia-nccl-cu13==2.30.x` wheel:
  `--env LD_PRELOAD=<lib> --env VLLM_NCCL_SO_PATH=<lib>`.

## Trainer layouts

`--layout` is a comma list of `dim=size` over `pp, dp, fsdp, tp, ep`; dimensions
left out are 1. Examples: `fsdp=2`, `pp=2`, `fsdp=2,ep=2`, `dp=2,fsdp=2`,
`fsdp=2,tp=2`. `--layout trainerless` runs the single-rank publisher
(`../hf_checkpoint_nccl_publisher.py`) as a baseline. That one reads the
checkpoint from disk inside the timed sync, so it is not comparable to the
other layouts.

How a rank's slice is chosen (`layout.py`):

- `dp`: `Replicate`. `fsdp`: `Shard(0)` on every tensor with ≥ 1 dimension.
- `tp`: by name. Column-parallel names (`q/k/v_proj`, `gate/up_proj`, `w1`,
  `w3`, embeddings, `lm_head`) get `Shard(0)`; row-parallel names (`o_proj`,
  `down_proj`, `w2`, ...) get `Shard(1)`; anything else is replicated. Add rules
  with `--placement-rules rules.json`, a JSON list of
  `[regex, "col"|"row"|"rep"]` checked before the defaults.
- `pp`: decoder layers split into contiguous stages. Embeddings go on the first
  stage; everything else outside the layers goes on the last.
- `ep`: a stage's `dp·fsdp` ranks form `ep` groups. Each group holds a
  contiguous block of `experts.<i>.` and shards it over its `edp` ranks the
  way fsdp would.
- Rank order is `pp` outermost, then `dp`, `fsdp`, `tp`.

Engines differ in what they accept. `engines.py:ENGINE_CAPS` records this, and
the driver refuses a layout the engine can't serve. The harness never works
around a missing capability (for example by gathering PP stages to rank 0 by
hand), because then it would be timing the harness. If an engine gains the
capability, flip its flag.

Other trainer flags:

- `--gather layer|tensor`: gather sharded weights with one collective per
  layer, as FSDP2 does (`layer`, the default), or one `full_tensor()` per
  tensor.
- `--stack-experts` (`nixl-slot-explore` only): hold each layer's experts as
  one `[E, ...]` DTensor per projection, like a grouped-GEMM trainer. Names on
  the wire stay per expert unless the engine stacks them. Not supported with
  `nccl_m2n`.

## Launchers and multi-node

- `--launcher torchrun` (the default) runs the trainer ranks under `torchrun`
  on `--trainer-gpus`.
- `--launcher ray` runs the same ranks as Ray actors in a placement group
  (`ray_launcher.py`), with `--ray-strategy PACK|SPREAD`. Without
  `--ray-address`, the driver starts a private Ray head that the server and
  trainer share, and stops it at the end. `sharded_rdt` must run this way.
- `--base-url http://host:port` attaches to a server you started yourself (any
  topology, any node) instead of launching one. You can't combine it with
  `--audit`. The server must have been started with
  `--weight-transfer-config '{"backend": "<engine>"}'` and
  `VLLM_SERVER_DEV_MODE=1`.
- To run one trainer rank by hand, use `trainer_main.py`, which takes the same
  flags (`--checkpoint` instead of `--update`). Use `--nccl-master-address` there
  if `get_ip()` picks the wrong network interface; `run_bench.py` doesn't
  forward it.
- `--env KEY=VALUE` (repeatable) is set in the server and in every trainer
  rank.

## Outputs

In `--output-dir`:

- `report.md`: a summary table, followed by the audit report if the audit ran.
  Paste it into a PR.
- `result.json`: the full record:
  - `summary`: `wall_seconds_median/min/max`, `gib_per_second_median`,
    `pause_window_seconds_median`, over the measured (non-warmup) syncs;
  - `iterations[]`: per sync, `wall_seconds`, `pause_window_seconds`,
    `rpc_seconds` (each HTTP call to the server with its time),
    `server_memory`, and per trainer rank `send_seconds` and
    `peak_transient_bytes`;
  - `trainer`: layout, `load_seconds`, `local_bytes_per_rank`;
  - `audit` (with `--audit`), plus versions, `serve_args` and the checkpoint
    manifest hash.
- `server.log`, the trainer logs, and the audit snapshots.
- One summary row is appended to `<output-dir>/../results.jsonl`, so the runs in
  one parent directory form a table.

Timing: `wall_seconds` is rank 0's `send_weights()` call from start to finish.
The pause window adds the server's `/pause` and `/resume` around it. Between
syncs the trainer reloads its shards from the next checkpoint without timing
it, as a training step would.

## The audit (`--audit`)

After the last sync, the driver snapshots every device tensor on the server
(parameters, buffers, and tensors held by kernels outside the registry). It
then starts a fresh server on the last checkpoint sent and compares the two.
Fresh snapshots are cached in `--fresh-cache-dir`, keyed by checkpoint files,
serve args and vLLM version, so rerunning on the same checkpoint skips the
second server. A run PASSes when:

- **equal to fresh:** every tensor has the fresh server's bytes and layout
  (`mismatched` = 0, nothing extra or missing);
- **stable:** no tensor that existed before the syncs moved address (`moved`
  = 0). CUDA graphs keep reading the old addresses;
- **no leak:** allocated memory grew by ≤ 64 MiB from the end of the first
  sync. Memory an engine allocates in the first sync and keeps, such as
  receive rings, is reported as "engine-resident" and isn't counted as growth.

`untested` lists tensors the update didn't change (KV scales, rotary caches,
...). A stale copy of one of those would still pass. Tensors that change on
their own between two snapshots (workspaces) count as runtime state and are
excluded. The server's `--gpu-memory-utilization` needs about 0.5 GiB of
headroom for the digest temporaries.

## Recipes that have been run (4× GB200, one node)

Qwen3-235B-A22B-FP8 (222.6 GiB), server TP=2 on GPUs 0,1, trainer on GPUs 2,3,
`--warmup 1 --iters 3 --audit --startup-timeout 3600`, with serve args
`--tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.8`.
Startup and the fresh server take most of the wall time; plan on 30–60 min per
run.

| `--backend` | `--layout` | launcher | median sync |
|---|---|---|---|
| `nccl` | `fsdp=2` | torchrun | 15.0 s (27.5 s with `--gather tensor`) |
| `sharded_rdt` | `fsdp=2` / `pp=2` / `fsdp=2,ep=2` | ray | 10.7 / 7.6 / 4.3–4.8 s |
| `nixl_slot` + `--stack-experts` | `fsdp=2` / `pp=2` / `fsdp=2,ep=2` | torchrun | 2.3 / 2.2 / 2.2 s |

Smaller smoke tests: `Qwen3-0.6B` (dense) or
`hmellor/tiny-random-DeepseekV2ForCausalLM` (MoE, exercises ep), with
`--max-model-len 2048 --gpu-memory-utilization 0.5`. These take a few minutes
each.

## Known problems

- **`sharded_rdt` segfaults** a server worker at the start of a sync, from a
  Ray 2.56.1 race (the RDT ref is registered after the task is submitted). It
  shows up mostly on small FP8 models and is not seen at 235B. Check
  `server.log` for `Segfault encountered`.
- **`sharded_rdt` needs memory headroom:** at `--gpu-memory-utilization 0.9`,
  the 235B server ran out of memory during RDT's background post-processing.
  0.8 works.
- **`sharded_rdt` with `--stack-experts`** is slower, and the 235B trainer ran
  out of memory. Use the per-expert trainer for RDT numbers.
- **FP8 MoE audits fail with ~376 "extra" kernel-held scale tensors** on vLLM
  builds whose reload keeps temporary scales in `moe_quant_config`. The values
  are correct; it's a vLLM bug, not a harness one.
- **First-start compile time:** FlashInfer's MoE autotune can take an hour on
  big MoE models. Add serve args
  `--kernel-config '{"enable_flashinfer_autotune": false}'` if you aren't
  timing the server's own startup.

## Files

| File | Purpose |
|---|---|
| `run_bench.py` | Driver: server (or `--base-url`), trainer, audit, `result.json`, `report.md`, `results.jsonl` |
| `trainer_main.py` | One trainer rank (torchrun entry point); timing and memory per sync |
| `ray_launcher.py` | The same ranks as Ray actors |
| `layout.py` | `TrainerLayout`, tp name rules, pp stage split |
| `sharded_checkpoint.py` | Per-rank checkpoint slices as DTensors; `GatherSource` and `ShardedM2NSource` |
| `engines.py` | `ENGINE_CAPS`, refusal reasons, per-engine server env/args and trainer init info |
| `../validation/` | `server.py`, `tensor_audit*.py`, `make_perturbed_checkpoint.py` |

To add an engine: add an `ENGINE_CAPS` entry, a `*_init_info` builder, any
server env/args it needs in `engines.py`, and a branch in
`trainer_main.run_sharded`. CPU tests for layouts and placement:
`pytest tests/test_bench_layout.py`.
