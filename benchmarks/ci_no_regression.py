#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI smoke no-regression gate for Rust euclidean fit timing and RSS")
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-ratio-threshold", type=float, default=1.35)
    p.add_argument("--rss-ratio-threshold", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=42)
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
    centers = np.stack([
        np.linspace(-2.0, 2.0, n_features, dtype=np.float32),
        np.linspace(2.0, -2.0, n_features, dtype=np.float32),
        np.zeros(n_features, dtype=np.float32),
    ])
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


def run_bench(binary: Path, data_path: Path, output_path: Path, time_path: Path, warmup: int, repeats: int, seed: int) -> Dict[str, float]:
    cmd = [
        "/usr/bin/time",
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
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "fit_mean_sec": float(payload["fit_mean_sec"]),
        "fit_std_sec": float(payload["fit_std_sec"]),
        "process_wall_sec": wall,
        "process_max_rss_mb": parse_rss_mb(time_path),
    }


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()

    candidate_bin = build_release(candidate_root)
    baseline_bin = build_release(baseline_root)

    with tempfile.TemporaryDirectory(prefix="umap-rs-ci-regression-") as tmpdir:
        tmp = Path(tmpdir)
        x = generate_dataset(args.seed)
        data_path = tmp / "dataset.csv"
        np.savetxt(data_path, x, delimiter=",", fmt="%.8f")

        baseline_metrics = run_bench(
            baseline_bin,
            data_path,
            tmp / "baseline_out.csv",
            tmp / "baseline.time.txt",
            args.warmup,
            args.repeats,
            args.seed,
        )
        candidate_metrics = run_bench(
            candidate_bin,
            data_path,
            tmp / "candidate_out.csv",
            tmp / "candidate.time.txt",
            args.warmup,
            args.repeats,
            args.seed,
        )

    time_ratio = candidate_metrics["fit_mean_sec"] / baseline_metrics["fit_mean_sec"]
    rss_ratio = candidate_metrics["process_max_rss_mb"] / baseline_metrics["process_max_rss_mb"]

    summary = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "ratios": {
            "fit_mean_sec": time_ratio,
            "process_max_rss_mb": rss_ratio,
        },
        "thresholds": {
            "fit_mean_sec": args.time_ratio_threshold,
            "process_max_rss_mb": args.rss_ratio_threshold,
        },
    }
    print(json.dumps(summary, indent=2))

    failures: List[str] = []
    if time_ratio > args.time_ratio_threshold:
        failures.append(
            f"candidate fit_mean_sec ratio {time_ratio:.4f} exceeded threshold {args.time_ratio_threshold:.4f}"
        )
    if rss_ratio > args.rss_ratio_threshold:
        failures.append(
            f"candidate process_max_rss_mb ratio {rss_ratio:.4f} exceeded threshold {args.rss_ratio_threshold:.4f}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
