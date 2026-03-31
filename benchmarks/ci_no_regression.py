#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

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


def resolve_time_bin() -> str:
    override = os.environ.get("UMAP_BENCH_TIME_BIN", "").strip()
    candidates: List[str] = []
    if override:
        candidates.append(override)
    candidates.extend(["/usr/bin/time", "gtime", "time"])

    for candidate in candidates:
        cmd = [candidate, "-v", "true"]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            continue
        out = f"{proc.stdout}\n{proc.stderr}"
        if "Maximum resident set size" in out:
            return candidate

    raise RuntimeError(
        "No suitable GNU time binary found. Set UMAP_BENCH_TIME_BIN to a GNU time executable "
        "that supports '-v' and reports 'Maximum resident set size'."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CI no-regression gate for Rust fit timing and RSS with randomized paired order "
            "and robust ratio statistics"
        )
    )
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--paired-runs", type=int, default=4)
    p.add_argument("--time-ratio-threshold", type=float, default=1.35)
    p.add_argument("--rss-ratio-threshold", type=float, default=1.25)
    p.add_argument("--tail-slack", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    return p.parse_args()


def build_release(root: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "fit_csv", "--bin", "bench_fit_csv"],
        cwd=root / "rust_umap",
        check=True,
    )
    return root / "rust_umap" / "target" / "release" / "bench_fit_csv"


def generate_dataset(seed: int, n_samples: int = 1400, n_features: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.stack(
        [
            np.linspace(-2.0, 2.0, n_features, dtype=np.float32),
            np.linspace(2.0, -2.0, n_features, dtype=np.float32),
            np.zeros(n_features, dtype=np.float32),
        ]
    )
    labels = rng.integers(0, len(centers), size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.4, size=(n_samples, n_features)).astype(np.float32)
    x = centers[labels] + noise
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def parse_rss_mb(time_file: Path) -> float:
    for line in time_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("Maximum resident set size (kbytes):"):
            return float(stripped.split(":", 1)[1].strip()) / 1024.0
    raise RuntimeError(f"failed to parse max RSS from {time_file}")


def run_bench(
    binary: Path,
    data_path: Path,
    output_path: Path,
    time_path: Path,
    warmup: int,
    repeats: int,
    seed: int,
    metric: str,
) -> Dict[str, float]:
    time_bin = resolve_time_bin()
    cmd = [
        time_bin,
        "-v",
        "-o",
        str(time_path),
        str(binary),
        str(data_path),
        str(output_path),
        "15",
        "2",
        "120",
        str(seed),
        "random",
        "false",
        "30",
        "10",
        "4096",
        str(warmup),
        str(repeats),
        "--metric",
        metric,
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "fit_mean_sec": float(payload["fit_mean_sec"]),
        "fit_std_sec": float(payload["fit_std_sec"]),
        "process_wall_sec": wall,
        "process_max_rss_mb": parse_rss_mb(time_path),
    }


def ratio_stats(candidate_vals: List[float], baseline_vals: List[float]) -> Dict[str, float]:
    c = np.asarray(candidate_vals, dtype=np.float64)
    b = np.asarray(baseline_vals, dtype=np.float64)
    if c.size != b.size:
        raise ValueError("candidate and baseline sample sizes must match for paired ratio stats")
    if c.size == 0:
        raise ValueError("empty sample set for ratio stats")
    if np.any(b <= 0.0):
        raise ValueError("baseline contains non-positive values, ratio undefined")

    ratio = c / b
    return {
        "median": float(np.median(ratio)),
        "p75": float(np.percentile(ratio, 75.0)),
        "mean": float(np.mean(ratio)),
        "std": float(np.std(ratio)),
        "samples": ratio.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.paired_runs < 1:
        raise SystemExit("--paired-runs must be >= 1")

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()

    candidate_bin = build_release(candidate_root)
    baseline_bin = build_release(baseline_root)

    rng = np.random.default_rng(args.seed ^ 0x5EEDC0DE)

    per_impl: Dict[str, List[Dict[str, float]]] = {"baseline": [], "candidate": []}
    paired_order: List[List[str]] = []

    with tempfile.TemporaryDirectory(prefix="umap-rs-ci-regression-") as tmpdir:
        tmp = Path(tmpdir)
        x = generate_dataset(args.seed)
        data_path = tmp / "dataset.csv"
        np.savetxt(data_path, x, delimiter=",", fmt="%.8f")

        for pair_idx in range(args.paired_runs):
            order = ["baseline", "candidate"]
            if int(rng.integers(0, 2)) == 1:
                order.reverse()
            paired_order.append(order)

            for impl in order:
                bin_path = baseline_bin if impl == "baseline" else candidate_bin
                run_metrics = run_bench(
                    bin_path,
                    data_path,
                    tmp / f"{impl}_pair{pair_idx}.out.csv",
                    tmp / f"{impl}_pair{pair_idx}.time.txt",
                    args.warmup,
                    args.repeats,
                    args.seed + pair_idx,
                    args.metric,
                )
                per_impl[impl].append(run_metrics)

    baseline_fit = [x["fit_mean_sec"] for x in per_impl["baseline"]]
    candidate_fit = [x["fit_mean_sec"] for x in per_impl["candidate"]]
    baseline_rss = [x["process_max_rss_mb"] for x in per_impl["baseline"]]
    candidate_rss = [x["process_max_rss_mb"] for x in per_impl["candidate"]]

    time_ratio = ratio_stats(candidate_fit, baseline_fit)
    rss_ratio = ratio_stats(candidate_rss, baseline_rss)

    summary = {
        "metric": args.metric,
        "paired_runs": args.paired_runs,
        "paired_order": paired_order,
        "candidate": per_impl["candidate"],
        "baseline": per_impl["baseline"],
        "ratios": {
            "fit_mean_sec": time_ratio,
            "process_max_rss_mb": rss_ratio,
        },
        "thresholds": {
            "fit_mean_sec_median": args.time_ratio_threshold,
            "fit_mean_sec_p75": args.time_ratio_threshold * (1.0 + args.tail_slack),
            "process_max_rss_mb_median": args.rss_ratio_threshold,
            "process_max_rss_mb_p75": args.rss_ratio_threshold * (1.0 + args.tail_slack),
        },
        "thread_env": THREAD_ENV,
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    print(json.dumps(summary, indent=2))

    failures: List[str] = []
    if time_ratio["median"] > args.time_ratio_threshold:
        failures.append(
            f"candidate fit_mean_sec median ratio {time_ratio['median']:.4f} exceeded threshold {args.time_ratio_threshold:.4f}"
        )
    if time_ratio["p75"] > args.time_ratio_threshold * (1.0 + args.tail_slack):
        failures.append(
            "candidate fit_mean_sec p75 ratio "
            f"{time_ratio['p75']:.4f} exceeded threshold {args.time_ratio_threshold * (1.0 + args.tail_slack):.4f}"
        )

    if rss_ratio["median"] > args.rss_ratio_threshold:
        failures.append(
            f"candidate process_max_rss_mb median ratio {rss_ratio['median']:.4f} exceeded threshold {args.rss_ratio_threshold:.4f}"
        )
    if rss_ratio["p75"] > args.rss_ratio_threshold * (1.0 + args.tail_slack):
        failures.append(
            "candidate process_max_rss_mb p75 ratio "
            f"{rss_ratio['p75']:.4f} exceeded threshold {args.rss_ratio_threshold * (1.0 + args.tail_slack):.4f}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
