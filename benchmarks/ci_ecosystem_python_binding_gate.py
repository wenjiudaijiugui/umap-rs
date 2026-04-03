#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


GROUP_E2E_MIXED = "e2e_mixed_knn_strategy"
GROUP_ALGO_EXACT = "algo_exact_shared_knn_exact"
LEGACY_GROUP_E2E = "e2e_default_ann"
LEGACY_GROUP_ALGO = "algo_exact_shared_knn"
PAIRWISE_KEY = "python_umap_learn__vs__umap_rs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Machine-readable gate for ecosystem python binding report: "
            "consistency + speed + memory"
        )
    )
    p.add_argument(
        "--report-json",
        default="benchmarks/report_ecosystem_python_binding.json",
        help="path to compare_ecosystem_python_binding.py JSON report",
    )
    p.add_argument("--max-trust-gap", type=float, default=0.03)
    p.add_argument("--max-recall-gap", type=float, default=0.06)
    p.add_argument("--min-knn-overlap", type=float, default=0.45)
    p.add_argument("--max-rust-over-python-speed-ratio", type=float, default=1.35)
    p.add_argument("--max-rust-over-python-memory-ratio", type=float, default=1.35)
    return p.parse_args()


def _as_float(x: object, field_name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not numeric: {x!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{field_name} is not finite: {x!r}")
    return v


def _positive_ratio(numer: float, denom: float, field_name: str) -> float:
    if denom <= 0.0:
        raise ValueError(f"{field_name} denominator must be > 0, got {denom}")
    return numer / denom


def validate_report_shape(report: Dict[str, object]) -> None:
    groups = report.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("report.groups must be an object")
    has_algo = GROUP_ALGO_EXACT in groups or LEGACY_GROUP_ALGO in groups
    has_e2e = GROUP_E2E_MIXED in groups or LEGACY_GROUP_E2E in groups
    if not has_algo:
        raise ValueError(
            f"missing group '{GROUP_ALGO_EXACT}' (or legacy '{LEGACY_GROUP_ALGO}') in report"
        )
    if not has_e2e:
        raise ValueError(
            f"missing group '{GROUP_E2E_MIXED}' (or legacy '{LEGACY_GROUP_E2E}') in report"
        )


def resolve_group_aliases(report: Dict[str, object]) -> Dict[str, str]:
    groups = report["groups"]
    algo_key = GROUP_ALGO_EXACT if GROUP_ALGO_EXACT in groups else LEGACY_GROUP_ALGO
    mixed_key = GROUP_E2E_MIXED if GROUP_E2E_MIXED in groups else LEGACY_GROUP_E2E
    return {
        "algo_key": algo_key,
        "mixed_key": mixed_key,
    }


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report_shape(report)
    aliases = resolve_group_aliases(report)

    exact_group = report["groups"][aliases["algo_key"]]
    mixed_group = report["groups"][aliases["mixed_key"]]

    failures: List[str] = []
    datasets_summary: Dict[str, object] = {}

    for dataset_name, dataset_payload in exact_group.items():
        fit = dataset_payload["fit_timing_and_process_memory"]
        consistency = dataset_payload["consistency"]

        py_fit = fit["python_umap_learn"]
        rust_fit = fit["umap_rs"]

        py_time = _as_float(py_fit["fit_mean_sec"], f"{dataset_name}.python.fit_mean_sec")
        rust_time = _as_float(rust_fit["fit_mean_sec"], f"{dataset_name}.rust.fit_mean_sec")
        py_rss = _as_float(py_fit["process_max_rss_mb"], f"{dataset_name}.python.process_max_rss_mb")
        rust_rss = _as_float(rust_fit["process_max_rss_mb"], f"{dataset_name}.rust.process_max_rss_mb")

        trust_py = _as_float(
            consistency["trustworthiness_at_15"]["python_umap_learn"],
            f"{dataset_name}.trust.python",
        )
        trust_rust = _as_float(
            consistency["trustworthiness_at_15"]["umap_rs"],
            f"{dataset_name}.trust.rust",
        )
        recall_py = _as_float(
            consistency["original_knn_recall_at_15"]["python_umap_learn"],
            f"{dataset_name}.recall.python",
        )
        recall_rust = _as_float(
            consistency["original_knn_recall_at_15"]["umap_rs"],
            f"{dataset_name}.recall.rust",
        )
        overlap = _as_float(
            consistency["pairwise"][PAIRWISE_KEY]["knn_overlap_at_15"],
            f"{dataset_name}.pairwise.knn_overlap_at_15",
        )

        trust_gap = abs(trust_rust - trust_py)
        recall_gap = abs(recall_rust - recall_py)
        speed_ratio = _positive_ratio(
            rust_time,
            py_time,
            f"{dataset_name}.speed_ratio_rust_over_python",
        )
        memory_ratio = _positive_ratio(
            rust_rss,
            py_rss,
            f"{dataset_name}.memory_ratio_rust_over_python",
        )

        checks = {
            "trust_gap_ok": trust_gap <= args.max_trust_gap,
            "recall_gap_ok": recall_gap <= args.max_recall_gap,
            "knn_overlap_ok": overlap >= args.min_knn_overlap,
            "speed_ratio_ok": speed_ratio <= args.max_rust_over_python_speed_ratio,
            "memory_ratio_ok": memory_ratio <= args.max_rust_over_python_memory_ratio,
        }

        if not checks["trust_gap_ok"]:
            failures.append(
                f"{dataset_name}: |trust_gap|={trust_gap:.6f} > {args.max_trust_gap:.6f}"
            )
        if not checks["recall_gap_ok"]:
            failures.append(
                f"{dataset_name}: |recall_gap|={recall_gap:.6f} > {args.max_recall_gap:.6f}"
            )
        if not checks["knn_overlap_ok"]:
            failures.append(
                f"{dataset_name}: knn_overlap@15={overlap:.6f} < {args.min_knn_overlap:.6f}"
            )
        if not checks["speed_ratio_ok"]:
            failures.append(
                f"{dataset_name}: rust/python fit_mean ratio={speed_ratio:.6f} > {args.max_rust_over_python_speed_ratio:.6f}"
            )
        if not checks["memory_ratio_ok"]:
            failures.append(
                f"{dataset_name}: rust/python RSS ratio={memory_ratio:.6f} > {args.max_rust_over_python_memory_ratio:.6f}"
            )

        datasets_summary[dataset_name] = {
            "trust_gap_abs": trust_gap,
            "recall_gap_abs": recall_gap,
            "knn_overlap_at_15": overlap,
            "speed_ratio_rust_over_python": speed_ratio,
            "memory_ratio_rust_over_python": memory_ratio,
            "checks": checks,
        }

    strategy_disclosure = {}
    if aliases["mixed_key"] == GROUP_E2E_MIXED:
        strategy_disclosure = {
            name: payload.get("knn_strategy", {}).get("equivalence", "missing")
            for name, payload in mixed_group.items()
        }
        for dataset_name, equivalence in strategy_disclosure.items():
            if equivalence not in {"strict_exact", "not_equivalent"}:
                failures.append(
                    f"{dataset_name}: group '{GROUP_E2E_MIXED}' missing explicit knn strategy equivalence tag"
                )
    else:
        strategy_disclosure = {name: "legacy_unavailable" for name in mixed_group.keys()}

    summary = {
        "report_json": str(report_path.resolve()),
        "group_checked_for_thresholds": GROUP_ALGO_EXACT,
        "group_key_resolved_for_thresholds": aliases["algo_key"],
        "mixed_group_key_resolved": aliases["mixed_key"],
        "legacy_group_alias_mode": aliases["mixed_key"] == LEGACY_GROUP_E2E
        or aliases["algo_key"] == LEGACY_GROUP_ALGO,
        "thresholds": {
            "max_trust_gap": args.max_trust_gap,
            "max_recall_gap": args.max_recall_gap,
            "min_knn_overlap": args.min_knn_overlap,
            "max_rust_over_python_speed_ratio": args.max_rust_over_python_speed_ratio,
            "max_rust_over_python_memory_ratio": args.max_rust_over_python_memory_ratio,
        },
        "datasets": datasets_summary,
        "strategy_disclosure_in_mixed_group": strategy_disclosure,
        "passed": len(failures) == 0,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
