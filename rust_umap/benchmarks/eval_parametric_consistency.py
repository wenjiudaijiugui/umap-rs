#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"
EXAMPLE_BIN = CRATE_DIR / "target" / "release" / "examples" / "parametric_eval"

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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Parametric UMAP MVP consistency/speed/memory against Python "
            "umap-learn ParametricUMAP (with automatic fallback proxy)."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["iris", "digits", "breast_cancer", "all"],
        default="all",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--umap-epochs", type=int, default=200)
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--consistency-tol", type=float, default=0.01)
    parser.add_argument("--output-json", default="", help="optional path to persist report JSON")

    # Worker mode for isolated Python RSS measurement.
    parser.add_argument("--worker", choices=["", "python_ref"], default="")
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--query-csv", default="")
    parser.add_argument("--train-out-csv", default="")
    parser.add_argument("--query-out-csv", default="")
    return parser.parse_args()


def load_dataset(name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if name == "iris":
        ds = load_iris()
        x, y = ds.data, ds.target
    elif name == "digits":
        ds = load_digits()
        x, y = ds.data, ds.target
    elif name == "breast_cancer":
        ds = load_breast_cancer()
        x, y = ds.data, ds.target
    else:
        raise ValueError(name)

    x = StandardScaler().fit_transform(x).astype(np.float32)
    return x, y


def parse_json_payload(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON payload found in process output")


def parse_max_rss_mb(text: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not match:
        return float("nan")
    return float(match.group(1)) / 1024.0


def read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def write_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.8f")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def knn_overlap(high: np.ndarray, low: np.ndarray, k: int) -> float:
    if high.shape[0] <= 2:
        return 1.0
    k = max(1, min(k, high.shape[0] - 1))

    neigh_high = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neigh_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    idx_high = neigh_high.fit(high).kneighbors(return_distance=False)[:, 1:]
    idx_low = neigh_low.fit(low).kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for row_high, row_low in zip(idx_high, idx_low):
        overlap = len(set(row_high.tolist()).intersection(set(row_low.tolist()))) / float(k)
        overlaps.append(overlap)
    return float(np.mean(overlaps))


def orthogonal_aligned_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for alignment: {a.shape} vs {b.shape}")

    a_centered = a - np.mean(a, axis=0, keepdims=True)
    b_centered = b - np.mean(b, axis=0, keepdims=True)

    a_norm = np.linalg.norm(a_centered)
    b_norm = np.linalg.norm(b_centered)
    if a_norm == 0.0 or b_norm == 0.0:
        return float("inf")

    a_unit = a_centered / a_norm
    b_unit = b_centered / b_norm

    u, _, vt = np.linalg.svd(b_unit.T @ a_unit, full_matrices=False)
    rotation = u @ vt
    b_aligned = b_unit @ rotation
    return rmse(a_unit, b_aligned)


def quality_metrics(x_query: np.ndarray, y_query: np.ndarray, k: int) -> Dict[str, float]:
    if y_query.ndim != 2 or y_query.shape[0] != x_query.shape[0]:
        raise ValueError(
            f"query embedding shape mismatch: x_query={x_query.shape}, y_query={y_query.shape}"
        )
    k = max(1, min(k, x_query.shape[0] - 1))
    return {
        "trustworthiness": float(trustworthiness(x_query, y_query, n_neighbors=k)),
        "knn_overlap": knn_overlap(x_query, y_query, k),
    }


def run_rust_impl(
    train: np.ndarray,
    query: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix=f"parametric-rust-{mode}-") as tmpdir:
        tmp = Path(tmpdir)
        train_csv = tmp / "train.csv"
        query_csv = tmp / "query.csv"
        train_out_csv = tmp / "train_out.csv"
        query_out_csv = tmp / "query_out.csv"
        time_txt = tmp / "time.txt"

        write_csv(train_csv, train)
        write_csv(query_csv, query)

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            str(EXAMPLE_BIN),
            str(train_csv),
            str(query_csv),
            str(train_out_csv),
            str(query_out_csv),
            str(args.seed),
            str(args.hidden_dim),
            str(args.train_epochs),
            str(args.batch_size),
            mode,
            str(args.n_neighbors),
            str(args.umap_epochs),
        ]

        env = os.environ.copy()
        env.update(THREAD_ENV)
        stdout = subprocess.check_output(cmd, text=True, env=env)
        payload = parse_json_payload(stdout)
        payload["process_max_rss_mb"] = parse_max_rss_mb(time_txt.read_text(encoding="utf-8"))
        return payload, read_csv(train_out_csv), read_csv(query_out_csv)


def run_python_worker(
    train: np.ndarray,
    query: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="parametric-python-ref-") as tmpdir:
        tmp = Path(tmpdir)
        train_csv = tmp / "train.csv"
        query_csv = tmp / "query.csv"
        train_out_csv = tmp / "train_out.csv"
        query_out_csv = tmp / "query_out.csv"
        time_txt = tmp / "time.txt"

        write_csv(train_csv, train)
        write_csv(query_csv, query)

        cmd = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_txt),
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "python_ref",
            "--train-csv",
            str(train_csv),
            "--query-csv",
            str(query_csv),
            "--train-out-csv",
            str(train_out_csv),
            "--query-out-csv",
            str(query_out_csv),
            "--seed",
            str(args.seed),
            "--hidden-dim",
            str(args.hidden_dim),
            "--train-epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--n-neighbors",
            str(args.n_neighbors),
            "--umap-epochs",
            str(args.umap_epochs),
        ]

        env = os.environ.copy()
        env.update(THREAD_ENV)
        stdout = subprocess.check_output(cmd, text=True, env=env)
        payload = parse_json_payload(stdout)
        payload["process_max_rss_mb"] = parse_max_rss_mb(time_txt.read_text(encoding="utf-8"))
        return payload, read_csv(train_out_csv), read_csv(query_out_csv)


def run_python_reference_impl(
    train: np.ndarray,
    query: np.ndarray,
    seed: int,
    hidden_dim: int,
    train_epochs: int,
    batch_size: int,
    n_neighbors: int,
    umap_epochs: int,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    np.random.seed(seed)
    random.seed(seed)

    backend = "umap_learn_parametric"
    fallback_reason = ""
    train_embedding: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None

    fit_time_sec = 0.0
    transform_query_time_sec = 0.0
    train_alignment_rmse = float("nan")
    train_alignment_mae = float("nan")

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from umap.parametric_umap import ParametricUMAP
        import tensorflow as tf

        tf.random.set_seed(seed)

        model = ParametricUMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            metric="euclidean",
            n_epochs=umap_epochs,
            batch_size=batch_size,
            random_state=seed,
            transform_seed=seed,
            verbose=False,
        )

        t0 = time.perf_counter()
        train_embedding = model.fit_transform(train).astype(np.float32)
        fit_time_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        query_embedding = model.transform(query).astype(np.float32)
        transform_query_time_sec = time.perf_counter() - t1

        train_pred = model.transform(train).astype(np.float32)
        train_alignment_rmse = rmse(train_pred, train_embedding)
        train_alignment_mae = mae(train_pred, train_embedding)
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"

        try:
            import umap
            from sklearn.neural_network import MLPRegressor

            backend = "proxy_umap_plus_mlp"

            t0 = time.perf_counter()
            teacher = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                metric="euclidean",
                n_epochs=umap_epochs,
                learning_rate=1.0,
                min_dist=0.1,
                spread=1.0,
                local_connectivity=1.0,
                set_op_mix_ratio=1.0,
                repulsion_strength=1.0,
                negative_sample_rate=5,
                random_state=seed,
                transform_seed=seed,
                low_memory=True,
                force_approximation_algorithm=False,
                n_jobs=1,
                init="spectral",
                verbose=False,
            ).fit_transform(train)

            reg = MLPRegressor(
                hidden_layer_sizes=(hidden_dim,),
                activation="tanh",
                solver="adam",
                alpha=1e-4,
                batch_size=batch_size,
                learning_rate_init=0.01,
                max_iter=train_epochs,
                random_state=seed,
                shuffle=True,
                tol=1e-5,
                n_iter_no_change=20,
            )
            reg.fit(train, teacher)
            train_embedding = reg.predict(train).astype(np.float32)
            fit_time_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            query_embedding = reg.predict(query).astype(np.float32)
            transform_query_time_sec = time.perf_counter() - t1

            train_alignment_rmse = rmse(train_embedding, teacher.astype(np.float32))
            train_alignment_mae = mae(train_embedding, teacher.astype(np.float32))
        except Exception as proxy_exc:
            from sklearn.decomposition import PCA
            from sklearn.neural_network import MLPRegressor

            backend = "proxy_pca_plus_mlp"
            fallback_reason = f"{fallback_reason} | proxy_umap_failed: {type(proxy_exc).__name__}: {proxy_exc}"

            t0 = time.perf_counter()
            teacher = PCA(n_components=2, random_state=seed).fit_transform(train).astype(np.float32)
            reg = MLPRegressor(
                hidden_layer_sizes=(hidden_dim,),
                activation="tanh",
                solver="adam",
                alpha=1e-4,
                batch_size=batch_size,
                learning_rate_init=0.01,
                max_iter=train_epochs,
                random_state=seed,
                shuffle=True,
                tol=1e-5,
                n_iter_no_change=20,
            )
            reg.fit(train, teacher)
            train_embedding = reg.predict(train).astype(np.float32)
            fit_time_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            query_embedding = reg.predict(query).astype(np.float32)
            transform_query_time_sec = time.perf_counter() - t1

            train_alignment_rmse = rmse(train_embedding, teacher)
            train_alignment_mae = mae(train_embedding, teacher)

    assert train_embedding is not None and query_embedding is not None

    payload = {
        "status": "ok",
        "backend": backend,
        "fallback_reason": fallback_reason,
        "fit_time_sec": float(fit_time_sec),
        "transform_query_time_sec": float(transform_query_time_sec),
        "train_alignment_rmse": float(train_alignment_rmse),
        "train_alignment_mae": float(train_alignment_mae),
        "n_train": int(train.shape[0]),
        "n_query": int(query.shape[0]),
    }
    return payload, train_embedding, query_embedding


def worker_python_ref(args: argparse.Namespace) -> None:
    if not args.train_csv or not args.query_csv:
        raise ValueError("--worker python_ref requires --train-csv and --query-csv")
    if not args.train_out_csv or not args.query_out_csv:
        raise ValueError("--worker python_ref requires --train-out-csv and --query-out-csv")

    train = read_csv(Path(args.train_csv))
    query = read_csv(Path(args.query_csv))

    payload, train_embedding, query_embedding = run_python_reference_impl(
        train=train,
        query=query,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        n_neighbors=args.n_neighbors,
        umap_epochs=args.umap_epochs,
    )

    write_csv(Path(args.train_out_csv), train_embedding)
    write_csv(Path(args.query_out_csv), query_embedding)
    print(json.dumps(payload))


def summarize_dataset(
    x_query: np.ndarray,
    rust_naive_payload: Dict[str, object],
    rust_naive_query_emb: np.ndarray,
    rust_opt_payload: Dict[str, object],
    rust_opt_query_emb: np.ndarray,
    py_payload: Dict[str, object],
    py_query_emb: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    quality_naive = quality_metrics(x_query, rust_naive_query_emb, args.metric_k)
    quality_opt = quality_metrics(x_query, rust_opt_query_emb, args.metric_k)
    quality_py = quality_metrics(x_query, py_query_emb, args.metric_k)

    fit_speedup = (
        float(rust_naive_payload["fit_time_sec"]) / float(rust_opt_payload["fit_time_sec"])
        if float(rust_opt_payload["fit_time_sec"]) > 0.0
        else float("inf")
    )
    transform_speedup = (
        float(rust_naive_payload["transform_query_time_sec"])
        / float(rust_opt_payload["transform_query_time_sec"])
        if float(rust_opt_payload["transform_query_time_sec"]) > 0.0
        else float("inf")
    )

    rust_naive_rss = float(rust_naive_payload["process_max_rss_mb"])
    rust_opt_rss = float(rust_opt_payload["process_max_rss_mb"])
    py_rss = float(py_payload["process_max_rss_mb"])

    memory_reduction_pct = (
        (1.0 - rust_opt_rss / rust_naive_rss) * 100.0 if rust_naive_rss > 0.0 else 0.0
    )
    rust_vs_python_rss_ratio = rust_opt_rss / py_rss if py_rss > 0.0 else float("inf")

    consistency = {
        "rust_opt_vs_python_aligned_rmse": orthogonal_aligned_rmse(
            rust_opt_query_emb, py_query_emb
        ),
        "rust_opt_minus_python_trustworthiness": (
            quality_opt["trustworthiness"] - quality_py["trustworthiness"]
        ),
        "rust_opt_minus_python_knn_overlap": (
            quality_opt["knn_overlap"] - quality_py["knn_overlap"]
        ),
        "bias_control": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "normalization": "StandardScaler applied before both Rust and Python pipelines",
            "metric_definitions": {
                "trustworthiness": f"sklearn.manifold.trustworthiness(k={args.metric_k})",
                "knn_overlap": f"kNN overlap @k={args.metric_k} on query set",
                "aligned_rmse": "orthogonal Procrustes-aligned RMSE",
            },
        },
    }

    gates = {
        "consistency_not_degraded_vs_rust_naive": (
            quality_opt["trustworthiness"] + args.consistency_tol >= quality_naive["trustworthiness"]
            and quality_opt["knn_overlap"] + args.consistency_tol >= quality_naive["knn_overlap"]
        ),
        "rust_optimized_faster_than_naive": (
            float(rust_opt_payload["fit_time_sec"]) <= float(rust_naive_payload["fit_time_sec"])
            and float(rust_opt_payload["transform_query_time_sec"])
            <= float(rust_naive_payload["transform_query_time_sec"])
        ),
        "rust_optimized_lower_or_equal_memory_than_naive": rust_opt_rss <= rust_naive_rss,
    }

    risks = []
    if py_payload.get("backend") != "umap_learn_parametric":
        risks.append(
            "Python side used fallback proxy; absolute parity with tensorflow-backed ParametricUMAP is limited."
        )
    if not gates["consistency_not_degraded_vs_rust_naive"]:
        risks.append("Rust optimized mode changed quality metrics beyond tolerance vs Rust naive mode.")

    return {
        "rust": {
            "naive": rust_naive_payload,
            "optimized": rust_opt_payload,
            "quality_query": {
                "naive": quality_naive,
                "optimized": quality_opt,
            },
        },
        "python_reference": {
            "run": py_payload,
            "quality_query": quality_py,
        },
        "consistency_review": consistency,
        "speed_review": {
            "fit_speedup_optimized_vs_naive": fit_speedup,
            "transform_speedup_optimized_vs_naive": transform_speedup,
            "rust_optimized_fit_vs_python_ratio": (
                float(rust_opt_payload["fit_time_sec"]) / float(py_payload["fit_time_sec"])
                if float(py_payload["fit_time_sec"]) > 0.0
                else float("inf")
            ),
            "rust_optimized_transform_vs_python_ratio": (
                float(rust_opt_payload["transform_query_time_sec"])
                / float(py_payload["transform_query_time_sec"])
                if float(py_payload["transform_query_time_sec"]) > 0.0
                else float("inf")
            ),
        },
        "memory_review": {
            "rust_naive_rss_mb": rust_naive_rss,
            "rust_optimized_rss_mb": rust_opt_rss,
            "python_rss_mb": py_rss,
            "rust_memory_reduction_pct_vs_naive": memory_reduction_pct,
            "rust_optimized_vs_python_rss_ratio": rust_vs_python_rss_ratio,
        },
        "gates": gates,
        "risks": risks,
    }


def orchestrate(args: argparse.Namespace) -> Dict[str, object]:
    datasets = ["iris", "digits", "breast_cancer"] if args.dataset == "all" else [args.dataset]
    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--example", "parametric_eval"],
        cwd=CRATE_DIR,
        check=True,
    )

    report: Dict[str, object] = {
        "config": {
            "datasets": datasets,
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "train_epochs": args.train_epochs,
            "batch_size": args.batch_size,
            "n_neighbors": args.n_neighbors,
            "umap_epochs": args.umap_epochs,
            "metric_k": args.metric_k,
            "thread_env": THREAD_ENV,
        },
        "datasets": {},
    }

    for name in datasets:
        x, y = load_dataset(name)
        stratify = y if y is not None and len(np.unique(y)) > 1 else None

        x_train, x_query = train_test_split(
            x,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
        x_train = np.asarray(x_train, dtype=np.float32)
        x_query = np.asarray(x_query, dtype=np.float32)

        rust_naive_payload, _, rust_naive_query_emb = run_rust_impl(
            train=x_train,
            query=x_query,
            mode="naive",
            args=args,
        )
        rust_opt_payload, _, rust_opt_query_emb = run_rust_impl(
            train=x_train,
            query=x_query,
            mode="optimized",
            args=args,
        )
        py_payload, _, py_query_emb = run_python_worker(
            train=x_train,
            query=x_query,
            args=args,
        )

        report["datasets"][name] = {
            "shape_train": list(x_train.shape),
            "shape_query": list(x_query.shape),
            **summarize_dataset(
                x_query=x_query,
                rust_naive_payload=rust_naive_payload,
                rust_naive_query_emb=rust_naive_query_emb,
                rust_opt_payload=rust_opt_payload,
                rust_opt_query_emb=rust_opt_query_emb,
                py_payload=py_payload,
                py_query_emb=py_query_emb,
                args=args,
            ),
        }

    return report


def main() -> None:
    args = parse_args()

    if args.worker == "python_ref":
        worker_python_ref(args)
        return

    report = orchestrate(args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
