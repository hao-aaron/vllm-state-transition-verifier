#!/usr/bin/env python3
"""Tensor-level audit of a weight update over the real transport.

``snapshot`` fetches a snapshot of every device tensor from a server that runs
``tensor_audit_extension.TensorAuditExtension`` (one entry per worker rank).
``compare`` checks one update step:

- correctness: every tensor of the updated server must have the same bytes
  as on a fresh server started on the same checkpoint (registered and
  off-registry, so kernel-held derived state is included);
- stability: no tensor that existed before the update may move (CUDA graphs
  keep reading the old address), appear or disappear;
- memory: allocated device memory may not grow by more than a threshold;
- coverage: tensors with the same bytes before the update and on the fresh
  server were not exercised by this update, and are listed as untested.

Runtime state (workspaces, caches written by requests) is told apart from
weights by snapshotting the same server twice with requests in between: a
tensor that changed without a weight update is runtime state and is excluded
from the correctness and coverage checks (and listed).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

KINDS = ("registered", "off_registry")
DEFAULT_MEMORY_GROWTH_LIMIT = 64 << 20

Snapshot = list[dict[str, Any]]  # one entry per worker rank


def fetch_snapshot(
    base_url: str, digests: bool = True, target: str = "target", timeout: int = 3600
) -> Snapshot:
    payload = {
        "method": "audit_snapshot",
        "args": ["1" if digests else "0", target],
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/collective_rpc",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        results = json.load(response)["results"]
    if not results or not all(isinstance(r, dict) for r in results):
        raise RuntimeError(
            "audit_snapshot returned no snapshot; is the server running with "
            "--worker-extension-cls tensor_audit_extension.TensorAuditExtension "
            f"and VLLM_SERVER_DEV_MODE=1? got: {str(results)[:200]}"
        )
    return sorted(results, key=lambda r: r.get("rank", 0))


def _ranks(*snapshots: Snapshot) -> int:
    counts = {len(s) for s in snapshots}
    if len(counts) != 1:
        raise ValueError(f"snapshots have different rank counts: {counts}")
    return counts.pop()


def runtime_names(first: Snapshot, second: Snapshot) -> dict[str, set[str]]:
    """Tensors whose bytes changed between two snapshots of the same server
    with no weight update in between (requests only)."""
    out = {kind: set() for kind in KINDS}
    for rank in range(_ranks(first, second)):
        for kind in KINDS:
            a, b = first[rank][kind], second[rank][kind]
            for name in a.keys() & b.keys():
                if a[name].get("digest") != b[name].get("digest"):
                    out[kind].add(name)
    return out


def _layout(entry: dict[str, Any]) -> tuple:
    return (
        tuple(entry.get("shape", ())),
        tuple(entry.get("stride", ())),
        entry.get("dtype"),
    )


def compare_step(
    before: Snapshot,
    after: Snapshot,
    fresh: Snapshot,
    runtime: dict[str, set[str]] | None = None,
    memory_growth_limit: int = DEFAULT_MEMORY_GROWTH_LIMIT,
) -> dict[str, Any]:
    """One update step: `before` and `after` come from the server under test,
    `fresh` from a server started on the checkpoint the update sent."""
    runtime = runtime or {kind: set() for kind in KINDS}
    findings: dict[str, Any] = {
        "mismatched": {k: [] for k in KINDS},
        "layout_mismatched": {k: [] for k in KINDS},
        "missing_vs_fresh": {k: [] for k in KINDS},
        "extra_vs_fresh": {k: [] for k in KINDS},
        "moved": {k: [] for k in KINDS},
        "disappeared": {k: [] for k in KINDS},
        "appeared": {k: [] for k in KINDS},
        "untested": {k: [] for k in KINDS},
        "runtime_excluded": {k: sorted(runtime[k]) for k in KINDS},
        "errors": {k: [] for k in KINDS},
        "memory": [],
    }
    compared = exercised = 0
    for rank in range(_ranks(before, after, fresh)):
        tag = f"rank{rank}:" if len(before) > 1 else ""
        for kind in KINDS:
            b, a, f = before[rank][kind], after[rank][kind], fresh[rank][kind]
            skip = runtime[kind]
            for name in sorted(a.keys() | f.keys()):
                if "error" in a.get(name, {}) or "error" in f.get(name, {}):
                    findings["errors"][kind].append(tag + name)
                    continue
                if name not in f:
                    findings["extra_vs_fresh"][kind].append(tag + name)
                    continue
                if name not in a:
                    findings["missing_vs_fresh"][kind].append(tag + name)
                    continue
                if _layout(a[name]) != _layout(f[name]):
                    findings["layout_mismatched"][kind].append(tag + name)
                if name in skip:
                    continue
                compared += 1
                if a[name].get("digest") != f[name].get("digest"):
                    findings["mismatched"][kind].append(tag + name)
                if name in b and b[name].get("digest") == f[name].get("digest"):
                    findings["untested"][kind].append(tag + name)
                else:
                    exercised += 1
            for name in sorted(b.keys() & a.keys()):
                if b[name].get("ptr") != a[name].get("ptr"):
                    findings["moved"][kind].append(tag + name)
            findings["disappeared"][kind] += [
                tag + n for n in sorted(b.keys() - a.keys())
            ]
            findings["appeared"][kind] += [tag + n for n in sorted(a.keys() - b.keys())]
        mb, ma = before[rank].get("memory", {}), after[rank].get("memory", {})
        if "allocated" in mb and "allocated" in ma:
            growth = ma["allocated"] - mb["allocated"]
            findings["memory"].append(
                {
                    "rank": rank,
                    "allocated_before": mb["allocated"],
                    "allocated_after": ma["allocated"],
                    "growth": growth,
                    "over_limit": growth > memory_growth_limit,
                }
            )
    failing = [
        key
        for key in (
            "mismatched",
            "layout_mismatched",
            "missing_vs_fresh",
            "extra_vs_fresh",
            "moved",
            "disappeared",
            "appeared",
            "errors",
        )
        if any(findings[key][k] for k in KINDS)
    ]
    if any(m["over_limit"] for m in findings["memory"]):
        failing.append("memory")
    counts = {
        key: sum(len(findings[key][k]) for k in KINDS)
        for key in findings
        if isinstance(findings[key], dict)
    }
    return {
        "status": "FAIL" if failing else "PASS",
        "failing_checks": failing,
        "counts": counts,
        # content-compared with the fresh server (runtime state excluded)
        "tensors_compared": compared,
        # ... of which the update changed the bytes (the rest are untested)
        "tensors_exercised": exercised,
        "findings": findings,
    }


def render_markdown(report: dict[str, Any], limit: int = 20) -> str:
    """A short summary to attach to a PR; the JSON report has every name."""
    lines = [f"## Weight-update tensor audit: {report['status']}", ""]
    for key in ("model", "server_args", "transport", "vllm_version"):
        if report.get(key):
            lines.append(f"- {key}: `{report[key]}`")
    lines += [
        "",
        "| step | status | compared | exercised | mismatched | moved | untested "
        "| memory growth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for step in report["steps"]:
        c = step["counts"]
        growth = max((m["growth"] for m in step["findings"]["memory"]), default=0)
        lines.append(
            f"| {step['name']} | {step['status']} | {step['tensors_compared']} | "
            f"{step['tensors_exercised']} | {c['mismatched']} | {c['moved']} | "
            f"{c['untested']} | {growth / 2**20:.1f} MiB |"
        )
    for step in report["steps"]:
        for key in step["failing_checks"] + ["untested", "runtime_excluded"]:
            if key == "memory":
                continue
            names = [n for k in KINDS for n in step["findings"][key][k]]
            if not names:
                continue
            lines += [
                "",
                f"<details><summary>{step['name']}: {key} ({len(names)})</summary>",
                "",
            ]
            lines += [f"- `{n}`" for n in names[:limit]]
            if len(names) > limit:
                lines.append(f"- ... {len(names) - limit} more in report.json")
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _load(path: Path) -> Snapshot:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--base-url", default="http://127.0.0.1:8000")
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--no-digests", action="store_true")
    snap.add_argument("--target", choices=("target", "draft"), default="target")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--before", type=Path, required=True)
    cmp.add_argument("--after", type=Path, required=True)
    cmp.add_argument("--fresh", type=Path, required=True)
    cmp.add_argument(
        "--runtime-pair",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        help="two snapshots of one server with only requests in between",
    )
    cmp.add_argument("--memory-growth-limit-mb", type=float, default=64)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "snapshot":
        result = fetch_snapshot(args.base_url, not args.no_digests, args.target)
        args.output.write_text(json.dumps(result) + "\n")
        print(json.dumps({"ranks": len(result), "output": str(args.output)}))
        return
    runtime = {kind: set() for kind in KINDS}
    for first, second in args.runtime_pair:
        for kind, names in runtime_names(_load(first), _load(second)).items():
            runtime[kind] |= names
    step = compare_step(
        _load(args.before),
        _load(args.after),
        _load(args.fresh),
        runtime,
        int(args.memory_growth_limit_mb * 2**20),
    )
    step["name"] = args.after.stem
    report = {"status": step["status"], "steps": [step]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(render_markdown(report))
    raise SystemExit(0 if step["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
