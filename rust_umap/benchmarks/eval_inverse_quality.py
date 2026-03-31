#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import umap
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Rust inverse_transform quality against umap-learn")
    p.add_argument(
        "--dataset",
        choices=["iris", "breast_cancer", "digits", "california_housing", "all"],
        default="all",
    )
    p.add_argument("--california-max-samples", type=int, default=3000)
    p.add_argument(
        "--allow-asymmetric-errors",
        action="store_true",
        help="do not fail when one implementation succeeds and the other fails",
    )
    return p.parse_args()


def load_dataset(name: str, california_max_samples: int) -> np.ndarray:
    if name == "breast_cancer":
        x = load_breast_cancer().data
    elif name == "iris":
        x = load_iris().data
    elif name == "digits":
        x = load_digits().data
    elif name == "california_housing":
        x = fetch_california_housing().data
        if x.shape[0] > california_max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(x.shape[0], size=california_max_samples, replace=False)
            x = x[idx]
    else:
        raise ValueError(name)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    return x


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _parse_json_payload(stdout: str) -> Dict[str, object]:
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
    raise ValueError("no JSON payload found in output")


def run_rust(train: np.ndarray, query: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory(prefix="umap-inverse-") as tmpdir:
        tmp = Path(tmpdir)
        train_csv = tmp / "train.csv"
        query_csv = tmp / "query.csv"
        np.savetxt(train_csv, train, delimiter=",", fmt="%.8f")
        np.savetxt(query_csv, query, delimiter=",", fmt="%.8f")

        cmd = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--example",
            "inverse_quality",
            "--manifest-path",
            str(CRATE_DIR / "Cargo.toml"),
            "--",
            str(train_csv),
            str(query_csv),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {
                "status": "error",
                "returncode": int(proc.returncode),
                "error": "rust inverse_quality example failed",
                "stderr": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
            }

        try:
            payload = _parse_json_payload(proc.stdout)
        except ValueError as exc:
            return {
                "status": "error",
                "returncode": 0,
                "error": f"failed to parse rust output JSON: {exc}",
                "stdout_tail": proc.stdout.strip().splitlines()[-3:],
            }

        payload["status"] = "ok"
        payload["train_inverse_status"] = "ok"
        payload["query_inverse_status"] = "ok"
        return payload


def run_python(train: np.ndarray, query: np.ndarray) -> dict:
    try:
        model = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            metric="euclidean",
            n_epochs=200,
            learning_rate=1.0,
            init="random",
            min_dist=0.1,
            spread=1.0,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            random_state=42,
            transform_seed=42,
            low_memory=True,
            force_approximation_algorithm=False,
            n_jobs=1,
            verbose=False,
        )

        t0 = time.perf_counter()
        train_embedding = model.fit_transform(train)
        fit_time_sec = time.perf_counter() - t0

        train_inverse_status = "ok"
        train_inverse_error = None
        train_reconstruction = None
        inverse_train_time_sec = None
        try:
            t1 = time.perf_counter()
            train_reconstruction = model.inverse_transform(train_embedding)
            inverse_train_time_sec = time.perf_counter() - t1
        except Exception as exc:  # pragma: no cover - benchmark fallback
            train_inverse_status = "error"
            train_inverse_error = f"{type(exc).__name__}: {exc}"

        t2 = time.perf_counter()
        query_embedding = model.transform(query)
        transform_query_time_sec = time.perf_counter() - t2

        query_inverse_status = "ok"
        query_inverse_error = None
        query_reconstruction = None
        inverse_query_time_sec = None
        try:
            t3 = time.perf_counter()
            query_reconstruction = model.inverse_transform(query_embedding)
            inverse_query_time_sec = time.perf_counter() - t3
        except Exception as exc:  # pragma: no cover - benchmark fallback
            query_inverse_status = "error"
            query_inverse_error = f"{type(exc).__name__}: {exc}"

        out = {
            "status": "ok",
            "fit_time_sec": fit_time_sec,
            "inverse_train_time_sec": inverse_train_time_sec,
            "transform_query_time_sec": transform_query_time_sec,
            "inverse_query_time_sec": inverse_query_time_sec,
            "train_inverse_status": train_inverse_status,
            "query_inverse_status": query_inverse_status,
        }
        if train_reconstruction is not None:
            out["train_rmse"] = rmse(train, train_reconstruction)
            out["train_mae"] = mae(train, train_reconstruction)
        if train_inverse_error is not None:
            out["train_inverse_error"] = train_inverse_error
        if query_reconstruction is not None:
            out["query_rmse"] = rmse(query, query_reconstruction)
            out["query_mae"] = mae(query, query_reconstruction)
        if query_inverse_error is not None:
            out["query_inverse_error"] = query_inverse_error
        return out
    except Exception as exc:  # pragma: no cover - benchmark fallback
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    names = ["iris", "breast_cancer", "digits", "california_housing"] if args.dataset == "all" else [args.dataset]

    report = {}
    failures = []
    for name in names:
        x = load_dataset(name, args.california_max_samples)
        train, query = train_test_split(x, test_size=0.2, random_state=42)
        rust = run_rust(train, query)
        py = run_python(train, query)

        status_block = {
            "impl_failure_asymmetry": (rust.get("status") == "ok") != (py.get("status") == "ok"),
            "train_inverse_asymmetry": rust.get("train_inverse_status", "ok") != py.get("train_inverse_status", "ok"),
            "query_inverse_asymmetry": rust.get("query_inverse_status", "ok") != py.get("query_inverse_status", "ok"),
        }

        entry = {
            "shape_train": list(train.shape),
            "shape_query": list(query.shape),
            "status": status_block,
            "rust": rust,
            "python_umap_learn": py,
        }

        rust_has_query_metrics = "query_rmse" in rust and "query_mae" in rust
        py_has_query_metrics = "query_rmse" in py and "query_mae" in py
        if rust_has_query_metrics and py_has_query_metrics:
            entry["delta_rust_minus_python"] = {
                key: float(rust[key] - py[key])
                for key in ["query_rmse", "query_mae"]
            }

        if status_block["impl_failure_asymmetry"]:
            failures.append(
                f"{name}: asymmetric implementation status rust={rust.get('status')} python={py.get('status')}"
            )
        if status_block["train_inverse_asymmetry"]:
            failures.append(
                f"{name}: asymmetric train inverse status rust={rust.get('train_inverse_status', 'ok')} python={py.get('train_inverse_status', 'ok')}"
            )
        if status_block["query_inverse_asymmetry"]:
            failures.append(
                f"{name}: asymmetric query inverse status rust={rust.get('query_inverse_status', 'ok')} python={py.get('query_inverse_status', 'ok')}"
            )

        report[name] = entry

    print(json.dumps(report, indent=2))

    if failures and not args.allow_asymmetric_errors:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
