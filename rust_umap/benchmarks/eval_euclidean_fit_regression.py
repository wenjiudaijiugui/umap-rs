#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"
BIN = CRATE_DIR / "target" / "release" / "bench_fit_csv"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Rust euclidean fit_transform timing/memory and enforce optional no-regression gates")
    p.add_argument("--dataset", choices=["digits", "california_housing", "all"], default="all")
    p.add_argument("--california-max-samples", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--baseline-report", default="", help="optional JSON baseline report path")
    p.add_argument("--time-ratio-threshold", type=float, default=1.20)
    p.add_argument("--rss-ratio-threshold", type=float, default=1.15)
    p.add_argument("--output-json", default="", help="optional path to persist current report")
    return p.parse_args()


def load_dataset(name: str, california_max_samples: int) -> np.ndarray:
    if name == "digits":
        x = load_digits().data
    elif name == "california_housing":
        x = fetch_california_housing().data
        if x.shape[0] > california_max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(x.shape[0], size=california_max_samples, replace=False)
            x = x[idx]
    else:
        raise ValueError(name)
    return StandardScaler().fit_transform(x).astype(np.float32)


def parse_max_rss_mb(text: str) -> float:
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not m:
        return float("nan")
    return float(m.group(1)) / 1024.0


def run_one(x: np.ndarray, warmup: int, repeats: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="umap-fit-") as tmpdir:
        tmp = Path(tmpdir)
        data_csv = tmp / "data.csv"
        out_csv = tmp / "out.csv"
        time_txt = tmp / "time.txt"
        np.savetxt(data_csv, x, delimiter=",", fmt="%.8f")

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            str(BIN),
            str(data_csv),
            str(out_csv),
            "15",
            "2",
            "200",
            "42",
            "random",
            "false",
            "30",
            "10",
            "4096",
            str(warmup),
            str(repeats),
            "--metric",
            "euclidean",
        ]
        env = os.environ.copy()
        env.update(THREAD_ENV)
        out = subprocess.check_output(cmd, text=True, env=env)
        payload = json.loads(out)
        rss_mb = parse_max_rss_mb(time_txt.read_text())
        return {
            "fit_mean_sec": float(payload["fit_mean_sec"]),
            "fit_std_sec": float(payload["fit_std_sec"]),
            "fit_times_sec": payload["fit_times_sec"],
            "process_max_rss_mb": rss_mb,
        }


def _dataset_block(report: Dict[str, object]) -> Dict[str, object]:
    if "datasets" in report and isinstance(report["datasets"], dict):
        return report["datasets"]
    return report


def compare_to_baseline(current: Dict[str, object], baseline: Dict[str, object], time_ratio_threshold: float, rss_ratio_threshold: float) -> Dict[str, object]:
    current_ds = _dataset_block(current)
    baseline_ds = _dataset_block(baseline)

    comparison = {}
    failures: List[str] = []

    for name, current_metrics in current_ds.items():
        if name not in baseline_ds:
            failures.append(f"{name}: missing in baseline report")
            continue

        baseline_metrics = baseline_ds[name]
        base_time = float(baseline_metrics["fit_mean_sec"])
        base_rss = float(baseline_metrics["process_max_rss_mb"])
        curr_time = float(current_metrics["fit_mean_sec"])
        curr_rss = float(current_metrics["process_max_rss_mb"])

        time_ratio = curr_time / base_time if base_time > 0 else float("inf")
        rss_ratio = curr_rss / base_rss if base_rss > 0 else float("inf")

        comparison[name] = {
            "baseline_fit_mean_sec": base_time,
            "current_fit_mean_sec": curr_time,
            "fit_mean_ratio": time_ratio,
            "baseline_process_max_rss_mb": base_rss,
            "current_process_max_rss_mb": curr_rss,
            "process_max_rss_ratio": rss_ratio,
        }

        if time_ratio > time_ratio_threshold:
            failures.append(
                f"{name}: fit_mean ratio {time_ratio:.4f} exceeded threshold {time_ratio_threshold:.4f}"
            )
        if rss_ratio > rss_ratio_threshold:
            failures.append(
                f"{name}: process_max_rss ratio {rss_ratio:.4f} exceeded threshold {rss_ratio_threshold:.4f}"
            )

    return {
        "comparison": comparison,
        "failures": failures,
        "thresholds": {
            "fit_mean_ratio": time_ratio_threshold,
            "process_max_rss_ratio": rss_ratio_threshold,
        },
    }


def main() -> None:
    args = parse_args()
    names = ["digits", "california_housing"] if args.dataset == "all" else [args.dataset]

    datasets = {}
    subprocess.run(["cargo", "build", "--release", "--quiet", "--bin", "bench_fit_csv"], cwd=CRATE_DIR, check=True)
    for name in names:
        x = load_dataset(name, args.california_max_samples)
        datasets[name] = run_one(x, args.warmup, args.repeats)
        datasets[name]["shape"] = list(x.shape)

    report: Dict[str, object] = {
        "config": {
            "metric": "euclidean",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "thread_env": THREAD_ENV,
        },
        "datasets": datasets,
    }

    if args.baseline_report:
        baseline_path = Path(args.baseline_report)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_cmp = compare_to_baseline(
            current=report,
            baseline=baseline,
            time_ratio_threshold=args.time_ratio_threshold,
            rss_ratio_threshold=args.rss_ratio_threshold,
        )
        report["baseline_comparison"] = baseline_cmp

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if args.baseline_report and report.get("baseline_comparison", {}).get("failures"):
        for failure in report["baseline_comparison"]["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
