#!/usr/bin/env python3
"""Wave-1 reliability smoke gate for algorithm tracks P1/P2/P3/P6/P7.

This gate is intentionally strict: any check failure exits non-zero.
It emits a machine-readable JSON summary for CI artifact review.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class CheckResult:
    id: str
    track: str
    pass_: bool
    elapsed_sec: float
    command: str
    stdout_tail: str
    stderr_tail: str


def _tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_check(check_id: str, track: str, cmd: List[str]) -> CheckResult:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return CheckResult(
        id=check_id,
        track=track,
        pass_=proc.returncode == 0,
        elapsed_sec=elapsed,
        command=" ".join(shlex.quote(tok) for tok in cmd),
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict Wave-1 algorithm smoke checks and emit JSON summary."
    )
    parser.add_argument(
        "--cargo-bin",
        default="cargo",
        help="cargo executable path (default: cargo)",
    )
    parser.add_argument(
        "--manifest-path",
        default="rust_umap/Cargo.toml",
        help="path to rust_umap Cargo manifest",
    )
    parser.add_argument(
        "--output-json",
        default="wave1-algo-smoke.json",
        help="output JSON report path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"manifest path not found: {manifest_path}")

    test_cases = [
        ("p1_input_contract_nonfinite", "P1", "tests::fit_rejects_non_finite_input_values"),
        ("p1_precomputed_distance_contract", "P1", "tests::precomputed_knn_rejects_invalid_distances"),
        ("p2_ann_recall_euclidean", "P2", "tests::approximate_knn_recall_reasonable_euclidean"),
        ("p2_ann_recall_cosine", "P2", "tests::approximate_knn_recall_reasonable_cosine"),
        ("p3_sparse_manhattan", "P3", "tests::sparse_csr_fit_transform_supports_manhattan_metric"),
        ("p3_sparse_cosine", "P3", "tests::sparse_csr_fit_transform_supports_cosine_metric"),
        (
            "p6_parametric_pairwise_validation",
            "P6",
            "parametric::tests::parametric_rejects_invalid_pairwise_loss_weight",
        ),
        (
            "p7_aligned_warmstart_gap",
            "P7",
            "aligned::tests::warmstart_reduces_initial_identity_gap",
        ),
    ]

    results: List[CheckResult] = []
    for check_id, track, test_name in test_cases:
        cmd = [
            args.cargo_bin,
            "test",
            "--manifest-path",
            str(manifest_path),
            test_name,
            "--",
            "--exact",
        ]
        res = run_check(check_id, track, cmd)
        results.append(res)

    overall_pass = all(r.pass_ for r in results)
    out = {
        "gate": "wave1_algo_smoke",
        "strict": True,
        "overall_pass": overall_pass,
        "timestamp_unix": int(time.time()),
        "checks": [
            {
                **{k if k != "pass_" else "pass": v for k, v in asdict(r).items()},
            }
            for r in results
        ],
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

