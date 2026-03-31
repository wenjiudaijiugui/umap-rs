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
from typing import Dict, List, Optional, Tuple

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

SHARED_UMAP_BASE_PARAMS = {
    "n_components": 2,
    "metric": "euclidean",
    "learning_rate": 1.0,
    "min_dist": 0.1,
    "spread": 1.0,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
    "repulsion_strength": 1.0,
    "negative_sample_rate": 5,
    "init": "spectral",
    "low_memory": True,
    "force_approximation_algorithm": False,
    "n_jobs": 1,
}

FALLBACK_MLP_PARAMS = {
    "activation": "tanh",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 0.01,
    "shuffle": True,
    "tol": 1e-5,
    "n_iter_no_change": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Parametric UMAP MVP consistency/speed/memory against Python "
            "umap-learn ParametricUMAP with explicit strict/fallback policy."
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
    parser.add_argument(
        "--python-ref-policy",
        choices=["strict", "fallback_aux"],
        default="strict",
        help=(
            "strict: require umap.parametric_umap.ParametricUMAP as Python reference; "
            "fallback_aux: allow fallback proxy for auxiliary-only diagnostics"
        ),
    )
    parser.add_argument(
        "--enforce-hard-gate",
        dest="enforce_hard_gate",
        action="store_true",
        default=True,
        help="exit non-zero if any dataset fails hard gates (default: enabled)",
    )
    parser.add_argument(
        "--no-enforce-hard-gate",
        dest="enforce_hard_gate",
        action="store_false",
        help="do not exit non-zero even when hard gate fails",
    )
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

    return np.asarray(x, dtype=np.float32), y


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


def aligned_umap_kwargs(n_neighbors: int, umap_epochs: int, seed: int) -> Dict[str, object]:
    kwargs = dict(SHARED_UMAP_BASE_PARAMS)
    kwargs.update(
        {
            "n_neighbors": n_neighbors,
            "n_epochs": umap_epochs,
            "random_state": seed,
            "transform_seed": seed,
        }
    )
    return kwargs


def parameter_alignment_report(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "shared_umap_hyperparams": aligned_umap_kwargs(
            n_neighbors=args.n_neighbors,
            umap_epochs=args.umap_epochs,
            seed=args.seed,
        ),
        "rust_parametric_network_hyperparams": {
            "hidden_dim": args.hidden_dim,
            "train_epochs": args.train_epochs,
            "batch_size": args.batch_size,
            "inference_batch_size": max(args.batch_size, 256),
            "learning_rate": FALLBACK_MLP_PARAMS["learning_rate_init"],
            "weight_decay": FALLBACK_MLP_PARAMS["alpha"],
            "standardize_input": False,
            "seed": args.seed,
        },
        "python_parametric_hyperparams": {
            "batch_size": args.batch_size,
            "tensorflow_seed": args.seed,
        },
        "python_fallback_proxy_hyperparams": {
            "mlp_hidden_layer_sizes": [args.hidden_dim],
            "mlp_max_iter": args.train_epochs,
            "mlp_batch_size": args.batch_size,
            **FALLBACK_MLP_PARAMS,
        },
        "notes": [
            "Primary Rust-vs-Python hard gate accepts only backend=umap_learn_parametric.",
            "Fallback backends are auxiliary diagnostics and never used in the primary hard gate.",
            "StandardScaler is fit on train split only, then reused for train/query transform.",
            "ParametricUMAP does not expose a direct one-to-one hidden_dim control in this harness.",
        ],
    }


def build_gate(status: str, reason: str, **details: object) -> Dict[str, object]:
    gate = {"status": status, "reason": reason}
    if details:
        gate["details"] = details
    return gate


def gate_status_from_bool(passed: bool, pass_reason: str, fail_reason: str, **details: object) -> Dict[str, object]:
    return build_gate("pass" if passed else "fail", pass_reason if passed else fail_reason, **details)


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
) -> Tuple[Dict[str, object], Optional[np.ndarray], Optional[np.ndarray]]:
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
            "--python-ref-policy",
            str(args.python_ref_policy),
        ]

        env = os.environ.copy()
        env.update(THREAD_ENV)
        stdout = subprocess.check_output(cmd, text=True, env=env)
        payload = parse_json_payload(stdout)
        payload["process_max_rss_mb"] = parse_max_rss_mb(time_txt.read_text(encoding="utf-8"))

        train_embedding = read_csv(train_out_csv) if payload.get("has_train_embedding") else None
        query_embedding = read_csv(query_out_csv) if payload.get("has_query_embedding") else None
        return payload, train_embedding, query_embedding


def run_python_reference_impl(
    train: np.ndarray,
    query: np.ndarray,
    seed: int,
    hidden_dim: int,
    train_epochs: int,
    batch_size: int,
    n_neighbors: int,
    umap_epochs: int,
    python_ref_policy: str,
) -> Tuple[Dict[str, object], Optional[np.ndarray], Optional[np.ndarray]]:
    np.random.seed(seed)
    random.seed(seed)

    umap_kwargs = aligned_umap_kwargs(
        n_neighbors=n_neighbors,
        umap_epochs=umap_epochs,
        seed=seed,
    )

    backend = "unavailable"
    status = "reference_unavailable"
    fallback_reason = ""
    train_embedding: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None
    used_fallback = False
    primary_reference_available = False
    comparable_for_primary_gate = False

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
            batch_size=batch_size,
            verbose=False,
            **umap_kwargs,
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
        status = "ok"
        backend = "umap_learn_parametric"
        primary_reference_available = True
        comparable_for_primary_gate = True
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        if python_ref_policy == "strict":
            payload = {
                "status": status,
                "backend": backend,
                "python_ref_policy": python_ref_policy,
                "reference_tier": "missing",
                "primary_reference_available": primary_reference_available,
                "comparable_for_primary_gate": comparable_for_primary_gate,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "fit_time_sec": float(fit_time_sec),
                "transform_query_time_sec": float(transform_query_time_sec),
                "train_alignment_rmse": float(train_alignment_rmse),
                "train_alignment_mae": float(train_alignment_mae),
                "has_train_embedding": False,
                "has_query_embedding": False,
                "n_train": int(train.shape[0]),
                "n_query": int(query.shape[0]),
            }
            return payload, None, None

        used_fallback = True
        try:
            import umap
            from sklearn.neural_network import MLPRegressor

            backend = "proxy_umap_plus_mlp"
            status = "auxiliary_fallback"

            t0 = time.perf_counter()
            teacher = umap.UMAP(verbose=False, **umap_kwargs).fit_transform(train)

            reg = MLPRegressor(
                hidden_layer_sizes=(hidden_dim,),
                batch_size=batch_size,
                max_iter=train_epochs,
                random_state=seed,
                **FALLBACK_MLP_PARAMS,
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
            status = "auxiliary_fallback"
            fallback_reason = f"{fallback_reason} | proxy_umap_failed: {type(proxy_exc).__name__}: {proxy_exc}"

            t0 = time.perf_counter()
            teacher = PCA(n_components=2, random_state=seed).fit_transform(train).astype(np.float32)
            reg = MLPRegressor(
                hidden_layer_sizes=(hidden_dim,),
                batch_size=batch_size,
                max_iter=train_epochs,
                random_state=seed,
                **FALLBACK_MLP_PARAMS,
            )
            reg.fit(train, teacher)
            train_embedding = reg.predict(train).astype(np.float32)
            fit_time_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            query_embedding = reg.predict(query).astype(np.float32)
            transform_query_time_sec = time.perf_counter() - t1

            train_alignment_rmse = rmse(train_embedding, teacher)
            train_alignment_mae = mae(train_embedding, teacher)

    has_train_embedding = train_embedding is not None
    has_query_embedding = query_embedding is not None

    payload = {
        "status": status,
        "backend": backend,
        "python_ref_policy": python_ref_policy,
        "reference_tier": "primary" if comparable_for_primary_gate else "auxiliary",
        "primary_reference_available": primary_reference_available,
        "comparable_for_primary_gate": comparable_for_primary_gate,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "fit_time_sec": float(fit_time_sec),
        "transform_query_time_sec": float(transform_query_time_sec),
        "train_alignment_rmse": float(train_alignment_rmse),
        "train_alignment_mae": float(train_alignment_mae),
        "has_train_embedding": has_train_embedding,
        "has_query_embedding": has_query_embedding,
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
        python_ref_policy=args.python_ref_policy,
    )

    if train_embedding is not None:
        write_csv(Path(args.train_out_csv), train_embedding)
    else:
        Path(args.train_out_csv).write_text("", encoding="utf-8")

    if query_embedding is not None:
        write_csv(Path(args.query_out_csv), query_embedding)
    else:
        Path(args.query_out_csv).write_text("", encoding="utf-8")

    print(json.dumps(payload))


def summarize_dataset(
    x_query: np.ndarray,
    rust_naive_payload: Dict[str, object],
    rust_naive_query_emb: np.ndarray,
    rust_opt_payload: Dict[str, object],
    rust_opt_query_emb: np.ndarray,
    py_payload: Dict[str, object],
    py_query_emb: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, object]:
    quality_naive = quality_metrics(x_query, rust_naive_query_emb, args.metric_k)
    quality_opt = quality_metrics(x_query, rust_opt_query_emb, args.metric_k)
    quality_py = quality_metrics(x_query, py_query_emb, args.metric_k) if py_query_emb is not None else None

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

    py_backend = str(py_payload.get("backend", "unavailable"))
    py_reference_is_primary = bool(py_payload.get("comparable_for_primary_gate", False))
    py_policy = str(py_payload.get("python_ref_policy", "strict"))

    primary_rmse: Optional[float] = None
    primary_trust_gap: Optional[float] = None
    primary_knn_gap: Optional[float] = None
    auxiliary_reference_review: Optional[Dict[str, object]] = None

    if quality_py is not None:
        aligned_rmse = orthogonal_aligned_rmse(rust_opt_query_emb, py_query_emb)
        trust_gap = quality_opt["trustworthiness"] - quality_py["trustworthiness"]
        knn_gap = quality_opt["knn_overlap"] - quality_py["knn_overlap"]
        if py_reference_is_primary:
            primary_rmse = aligned_rmse
            primary_trust_gap = trust_gap
            primary_knn_gap = knn_gap
        else:
            auxiliary_reference_review = {
                "note": (
                    "Auxiliary fallback reference diagnostics only; "
                    "excluded from primary Rust-vs-Python hard gate."
                ),
                "backend": py_backend,
                "rust_opt_vs_aux_aligned_rmse": aligned_rmse,
                "rust_opt_minus_aux_trustworthiness": trust_gap,
                "rust_opt_minus_aux_knn_overlap": knn_gap,
            }

    consistency = {
        "primary_reference_policy": py_policy,
        "primary_reference_backend": py_backend,
        "primary_reference_comparable": py_reference_is_primary,
        "rust_opt_vs_python_aligned_rmse": primary_rmse,
        "rust_opt_minus_python_trustworthiness": primary_trust_gap,
        "rust_opt_minus_python_knn_overlap": primary_knn_gap,
        "bias_control": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "normalization": "StandardScaler fit on train split only; train/query transformed with same scaler",
            "metric_definitions": {
                "trustworthiness": f"sklearn.manifold.trustworthiness(k={args.metric_k})",
                "knn_overlap": f"kNN overlap @k={args.metric_k} on query set",
                "aligned_rmse": "orthogonal Procrustes-aligned RMSE",
            },
        },
    }

    gate_consistency_vs_naive = (
        quality_opt["trustworthiness"] + args.consistency_tol >= quality_naive["trustworthiness"]
        and quality_opt["knn_overlap"] + args.consistency_tol >= quality_naive["knn_overlap"]
    )
    gate_faster_vs_naive = (
        float(rust_opt_payload["fit_time_sec"]) <= float(rust_naive_payload["fit_time_sec"])
        and float(rust_opt_payload["transform_query_time_sec"])
        <= float(rust_naive_payload["transform_query_time_sec"])
    )
    gate_memory_vs_naive = rust_opt_rss <= rust_naive_rss

    gates: Dict[str, Dict[str, object]] = {
        "consistency_not_degraded_vs_rust_naive": gate_status_from_bool(
            gate_consistency_vs_naive,
            "Rust optimized quality stays within tolerance vs Rust naive.",
            "Rust optimized quality exceeds tolerance drift vs Rust naive.",
            tolerance=args.consistency_tol,
            rust_opt_quality=quality_opt,
            rust_naive_quality=quality_naive,
        ),
        "rust_optimized_faster_than_naive": gate_status_from_bool(
            gate_faster_vs_naive,
            "Rust optimized is faster or equal on fit+query transform vs Rust naive.",
            "Rust optimized is slower than Rust naive on fit or query transform.",
            rust_opt_fit_sec=float(rust_opt_payload["fit_time_sec"]),
            rust_naive_fit_sec=float(rust_naive_payload["fit_time_sec"]),
            rust_opt_transform_sec=float(rust_opt_payload["transform_query_time_sec"]),
            rust_naive_transform_sec=float(rust_naive_payload["transform_query_time_sec"]),
        ),
        "rust_optimized_lower_or_equal_memory_than_naive": gate_status_from_bool(
            gate_memory_vs_naive,
            "Rust optimized uses lower or equal RSS vs Rust naive.",
            "Rust optimized uses higher RSS than Rust naive.",
            rust_opt_rss_mb=rust_opt_rss,
            rust_naive_rss_mb=rust_naive_rss,
        ),
    }

    if py_reference_is_primary and quality_py is not None:
        gate_rust_vs_python_quality = (
            quality_opt["trustworthiness"] + args.consistency_tol >= quality_py["trustworthiness"]
            and quality_opt["knn_overlap"] + args.consistency_tol >= quality_py["knn_overlap"]
        )
        gates["python_reference_is_parametric_primary"] = build_gate(
            "pass",
            "Python reference uses umap.parametric_umap.ParametricUMAP.",
            backend=py_backend,
            policy=py_policy,
        )
        gates["rust_optimized_not_degraded_vs_python_primary"] = gate_status_from_bool(
            gate_rust_vs_python_quality,
            "Rust optimized quality stays within tolerance vs Python ParametricUMAP primary reference.",
            "Rust optimized quality exceeds tolerance vs Python ParametricUMAP primary reference.",
            tolerance=args.consistency_tol,
            rust_opt_quality=quality_opt,
            python_primary_quality=quality_py,
        )
        gates["rust_vs_python_hard_gate"] = gate_status_from_bool(
            gate_rust_vs_python_quality,
            "Rust-vs-Python hard gate passed.",
            "Rust-vs-Python hard gate failed on quality deltas.",
            backend=py_backend,
            policy=py_policy,
        )
    else:
        unavailable_reason = (
            "Python primary reference unavailable: "
            f"status={py_payload.get('status')}, backend={py_backend}, policy={py_policy}."
        )
        gates["python_reference_is_parametric_primary"] = build_gate(
            "fail",
            unavailable_reason,
        )
        gates["rust_optimized_not_degraded_vs_python_primary"] = build_gate(
            "skipped",
            "Skipped because Python primary reference gate failed.",
        )
        gates["rust_vs_python_hard_gate"] = build_gate(
            "fail",
            "Rust-vs-Python hard gate failed because Python primary reference is unavailable.",
            unavailable_reason=unavailable_reason,
        )

    hard_gate_keys = [
        "python_reference_is_parametric_primary",
        "rust_vs_python_hard_gate",
    ]
    failed_hard_gates = [name for name in hard_gate_keys if gates[name]["status"] == "fail"]
    failed_all_gates = [name for name, gate in gates.items() if gate["status"] == "fail"]
    skipped_gates = [name for name, gate in gates.items() if gate["status"] == "skipped"]

    risks: List[str] = []
    if not py_reference_is_primary:
        risks.append(
            "Python side is not a primary ParametricUMAP reference; Rust-vs-Python hard gate cannot be considered passed."
        )
    if not gate_consistency_vs_naive:
        risks.append(
            "Rust optimized mode changed quality metrics beyond tolerance vs Rust naive mode."
        )
    if failed_hard_gates:
        risks.append(f"Hard gate failed: {', '.join(failed_hard_gates)}")

    python_quality_section: Dict[str, object] = {"run": py_payload, "quality_query": quality_py}
    if auxiliary_reference_review is not None:
        python_quality_section["auxiliary_reference_review"] = auxiliary_reference_review

    return {
        "rust": {
            "naive": rust_naive_payload,
            "optimized": rust_opt_payload,
            "quality_query": {
                "naive": quality_naive,
                "optimized": quality_opt,
            },
        },
        "python_reference": python_quality_section,
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
        "gate_summary": {
            "overall_status": "pass" if not failed_all_gates else "fail",
            "hard_gate_status": "pass" if not failed_hard_gates else "fail",
            "failed_gates": failed_all_gates,
            "failed_hard_gates": failed_hard_gates,
            "skipped_gates": skipped_gates,
        },
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
            "python_ref_policy": args.python_ref_policy,
            "enforce_hard_gate": args.enforce_hard_gate,
            "thread_env": THREAD_ENV,
            "parameter_alignment": parameter_alignment_report(args),
        },
        "datasets": {},
    }

    for name in datasets:
        x, y = load_dataset(name)
        stratify = y if y is not None and len(np.unique(y)) > 1 else None

        x_train_raw, x_query_raw = train_test_split(
            x,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
        scaler = StandardScaler().fit(x_train_raw)
        x_train = scaler.transform(x_train_raw).astype(np.float32)
        x_query = scaler.transform(x_query_raw).astype(np.float32)

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

    hard_gate_fail_datasets: List[str] = []
    overall_fail_datasets: List[str] = []
    for name, result in report["datasets"].items():
        gate_summary = result.get("gate_summary", {})
        if gate_summary.get("hard_gate_status") != "pass":
            hard_gate_fail_datasets.append(name)
        if gate_summary.get("overall_status") != "pass":
            overall_fail_datasets.append(name)

    report["gate_overview"] = {
        "overall_status": "pass" if not overall_fail_datasets else "fail",
        "hard_gate_status": "pass" if not hard_gate_fail_datasets else "fail",
        "failed_datasets": overall_fail_datasets,
        "failed_hard_gate_datasets": hard_gate_fail_datasets,
        "status_line": (
            "PASS"
            if not hard_gate_fail_datasets
            else f"FAIL: hard gate failed on datasets={hard_gate_fail_datasets}"
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

    gate_overview = report.get("gate_overview", {})
    status_line = str(gate_overview.get("status_line", "UNKNOWN"))
    print(f"[gate] {status_line}", file=sys.stderr)

    if args.enforce_hard_gate and gate_overview.get("hard_gate_status") != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
