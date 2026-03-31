#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import umap
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"
BENCH_BIN = CRATE_DIR / "target" / "release" / "bench_fit_csv"

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
    p = argparse.ArgumentParser(
        description="Benchmark sparse CSR fit path (Rust vs umap-learn) on consistency/speed/memory"
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", "digits_sparse", "all"],
        default="all",
        help="datasets to benchmark",
    )
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--synthetic-samples", type=int, default=1400)
    p.add_argument("--synthetic-features", type=int, default=1800)
    p.add_argument("--synthetic-density", type=float, default=0.01)
    p.add_argument(
        "--max-consistency-samples",
        type=int,
        default=800,
        help="subsample size for pairwise-distance consistency metrics",
    )
    p.add_argument("--output-json", default="", help="optional report output path")

    # Internal-only mode: run umap-learn in a subprocess so /usr/bin/time can isolate RSS.
    p.add_argument("--internal-python-fit", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--input-npz", default="", help=argparse.SUPPRESS)
    p.add_argument("--output-csv", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def parse_max_rss_mb(text: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not match:
        return float("nan")
    return float(match.group(1)) / 1024.0


def write_csv_vector(path: Path, values: np.ndarray, fmt: str) -> None:
    arr = np.asarray(values).reshape(1, -1)
    np.savetxt(path, arr, delimiter=",", fmt=fmt)


def load_embeddings(path: Path) -> np.ndarray:
    emb = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb


def make_synthetic_sparse(
    n_samples: int, n_features: int, density: float, seed: int
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)

    def _data_rvs(k: int) -> np.ndarray:
        return rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)

    x = sp.random(
        n_samples,
        n_features,
        density=density,
        format="csr",
        random_state=seed,
        data_rvs=_data_rvs,
    ).astype(np.float32)
    x.sum_duplicates()
    x.sort_indices()
    return x


def load_dataset(name: str, args: argparse.Namespace) -> sp.csr_matrix:
    if name == "synthetic":
        return make_synthetic_sparse(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            density=args.synthetic_density,
            seed=args.seed,
        )
    if name == "digits_sparse":
        x = load_digits().data.astype(np.float32)
        x = sp.csr_matrix(x)
        x.sum_duplicates()
        x.sort_indices()
        return x
    raise ValueError(f"unsupported dataset {name}")


def run_rust_sparse(
    x: sp.csr_matrix, args: argparse.Namespace
) -> Tuple[Dict[str, object], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="umap-rs-sparse-rust-") as tmpdir:
        tmp = Path(tmpdir)
        dummy_input = tmp / "dummy.csv"
        out_csv = tmp / "rust_embedding.csv"
        indptr_csv = tmp / "indptr.csv"
        indices_csv = tmp / "indices.csv"
        values_csv = tmp / "values.csv"
        time_txt = tmp / "rust_time.txt"

        dummy_input.write_text("0\n", encoding="utf-8")
        write_csv_vector(indptr_csv, x.indptr, "%d")
        write_csv_vector(indices_csv, x.indices, "%d")
        write_csv_vector(values_csv, x.data.astype(np.float32), "%.8f")

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            str(BENCH_BIN),
            str(dummy_input),
            str(out_csv),
            str(args.n_neighbors),
            str(args.n_components),
            str(args.n_epochs),
            str(args.seed),
            "random",
            "false",
            "30",
            "10",
            "4096",
            str(args.warmup),
            str(args.repeats),
            "--metric",
            "euclidean",
            "--csr-indptr",
            str(indptr_csv),
            "--csr-indices",
            str(indices_csv),
            "--csr-data",
            str(values_csv),
            "--csr-n-cols",
            str(x.shape[1]),
        ]
        env = os.environ.copy()
        env.update(THREAD_ENV)
        stdout = subprocess.check_output(cmd, text=True, env=env)
        payload = json.loads(stdout)
        payload["process_max_rss_mb"] = parse_max_rss_mb(time_txt.read_text(encoding="utf-8"))
        embedding = load_embeddings(out_csv)
        return payload, embedding


def run_python_internal(args: argparse.Namespace) -> None:
    x = sp.load_npz(args.input_npz).tocsr().astype(np.float32)
    model = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric="euclidean",
        n_epochs=args.n_epochs,
        learning_rate=1.0,
        init="random",
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
        force_approximation_algorithm=False,
        n_jobs=1,
        verbose=False,
    )

    t0 = time.perf_counter()
    embedding = model.fit_transform(x)
    fit_sec = time.perf_counter() - t0

    np.savetxt(args.output_csv, embedding.astype(np.float32), delimiter=",", fmt="%.8f")
    print(json.dumps({"fit_sec": fit_sec}))


def run_python_sparse(
    x: sp.csr_matrix, args: argparse.Namespace
) -> Tuple[Dict[str, object], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="umap-rs-sparse-py-") as tmpdir:
        tmp = Path(tmpdir)
        input_npz = tmp / "input_sparse.npz"
        output_csv = tmp / "py_embedding.csv"
        time_txt = tmp / "py_time.txt"
        sp.save_npz(input_npz, x)

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            sys.executable,
            str(__file__),
            "--internal-python-fit",
            "--input-npz",
            str(input_npz),
            "--output-csv",
            str(output_csv),
            "--n-neighbors",
            str(args.n_neighbors),
            "--n-components",
            str(args.n_components),
            "--n-epochs",
            str(args.n_epochs),
            "--seed",
            str(args.seed),
        ]
        env = os.environ.copy()
        env.update(THREAD_ENV)
        stdout = subprocess.check_output(cmd, text=True, env=env)
        payload = json.loads(stdout)
        payload["process_max_rss_mb"] = parse_max_rss_mb(time_txt.read_text(encoding="utf-8"))
        embedding = load_embeddings(output_csv)
        return payload, embedding


def procrustes_rmse(rust_emb: np.ndarray, py_emb: np.ndarray) -> float:
    rust = rust_emb - rust_emb.mean(axis=0, keepdims=True)
    py = py_emb - py_emb.mean(axis=0, keepdims=True)
    rust /= np.linalg.norm(rust) + 1e-12
    py /= np.linalg.norm(py) + 1e-12
    rotation, _ = orthogonal_procrustes(py, rust)
    py_aligned = py @ rotation
    return float(np.sqrt(np.mean((rust - py_aligned) ** 2)))


def pairwise_distance_corr(
    rust_emb: np.ndarray, py_emb: np.ndarray, max_samples: int, seed: int
) -> float:
    n = rust_emb.shape[0]
    if n <= 2:
        return 1.0
    rng = np.random.default_rng(seed)
    m = min(n, max_samples)
    idx = rng.choice(n, size=m, replace=False)
    rust_d = pdist(rust_emb[idx], metric="euclidean")
    py_d = pdist(py_emb[idx], metric="euclidean")
    if rust_d.size == 0 or py_d.size == 0:
        return 1.0
    corr = np.corrcoef(rust_d, py_d)[0, 1]
    return float(corr)


def evaluate_dataset(name: str, x: sp.csr_matrix, args: argparse.Namespace) -> Dict[str, object]:
    rust_payload, rust_embedding = run_rust_sparse(x, args)
    py_payload, py_embedding = run_python_sparse(x, args)

    rust_trust = trustworthiness(x, rust_embedding, n_neighbors=args.n_neighbors)
    py_trust = trustworthiness(x, py_embedding, n_neighbors=args.n_neighbors)
    embed_corr = pairwise_distance_corr(
        rust_embedding, py_embedding, args.max_consistency_samples, args.seed
    )
    align_rmse = procrustes_rmse(rust_embedding, py_embedding)

    rust_mean = float(rust_payload["fit_mean_sec"])
    py_time = float(py_payload["fit_sec"])
    rust_rss = float(rust_payload["process_max_rss_mb"])
    py_rss = float(py_payload["process_max_rss_mb"])

    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])),
        "consistency": {
            "trustworthiness_rust": float(rust_trust),
            "trustworthiness_python": float(py_trust),
            "trustworthiness_delta_rust_minus_python": float(rust_trust - py_trust),
            "embedding_pairwise_distance_corr": embed_corr,
            "procrustes_rmse": align_rmse,
        },
        "speed": {
            "rust_fit_mean_sec": rust_mean,
            "python_fit_sec": py_time,
            "python_over_rust_ratio": float(py_time / rust_mean) if rust_mean > 0 else float("inf"),
        },
        "memory": {
            "rust_process_max_rss_mb": rust_rss,
            "python_process_max_rss_mb": py_rss,
            "python_over_rust_ratio": float(py_rss / rust_rss) if rust_rss > 0 else float("inf"),
        },
        "rust_raw": rust_payload,
        "python_raw": py_payload,
    }


def main() -> None:
    args = parse_args()

    if args.internal_python_fit:
        run_python_internal(args)
        return

    datasets = ["synthetic", "digits_sparse"] if args.dataset == "all" else [args.dataset]

    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--bin", "bench_fit_csv"],
        cwd=CRATE_DIR,
        check=True,
    )

    report: Dict[str, object] = {
        "config": {
            "metric": "euclidean",
            "n_neighbors": args.n_neighbors,
            "n_components": args.n_components,
            "n_epochs": args.n_epochs,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "thread_env": THREAD_ENV,
        },
        "datasets": {},
    }

    for name in datasets:
        x = load_dataset(name, args)
        report["datasets"][name] = evaluate_dataset(name, x, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
