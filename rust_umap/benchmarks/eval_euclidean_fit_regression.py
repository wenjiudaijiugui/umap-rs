#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Measure Rust euclidean fit_transform timing and memory")
    p.add_argument("--dataset", choices=["digits", "california_housing", "all"], default="all")
    p.add_argument("--california-max-samples", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
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


def main() -> None:
    args = parse_args()
    names = ["digits", "california_housing"] if args.dataset == "all" else [args.dataset]
    report = {}
    subprocess.run(["cargo", "build", "--release", "--quiet", "--bin", "bench_fit_csv"], cwd=CRATE_DIR, check=True)
    for name in names:
        x = load_dataset(name, args.california_max_samples)
        report[name] = run_one(x, args.warmup, args.repeats)
        report[name]["shape"] = list(x.shape)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
