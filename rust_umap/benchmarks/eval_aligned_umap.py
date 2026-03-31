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
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CRATE_DIR = ROOT / "rust_umap"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rust AlignedUMAP MVP against umap-learn AlignedUMAP"
    )
    parser.add_argument("--worker-python", action="store_true")
    parser.add_argument("--slice-csv", action="append", default=[])
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--drift", type=float, default=0.18)
    parser.add_argument("--noise", type=float, default=0.03)

    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alignment-regularization", type=float, default=0.08)
    parser.add_argument("--alignment-learning-rate", type=float, default=0.25)
    parser.add_argument("--alignment-epochs", type=int, default=100)
    parser.add_argument("--recenter-interval", type=int, default=5)

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consistency-sample", type=int, default=256)
    parser.add_argument("--output-json", default="")

    return parser.parse_args()


def save_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr.astype(np.float32), delimiter=",", fmt="%.8f")


def load_csv(path: Path) -> np.ndarray:
    out = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


def synthetic_temporal_slices(
    n_slices: int,
    n_samples: int,
    n_features: int,
    drift: float,
    noise: float,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False, dtype=np.float32)

    latent_base = np.stack(
        [
            np.cos(theta),
            np.sin(theta),
            np.cos(2.0 * theta) * 0.7,
            np.sin(3.0 * theta) * 0.4,
        ],
        axis=1,
    )

    projection = rng.normal(0.0, 1.0, size=(latent_base.shape[1], n_features)).astype(np.float32)

    slices: List[np.ndarray] = []
    for slice_idx in range(n_slices):
        phase = drift * float(slice_idx)
        rot = np.array(
            [
                [np.cos(phase), -np.sin(phase)],
                [np.sin(phase), np.cos(phase)],
            ],
            dtype=np.float32,
        )

        latent = latent_base.copy()
        latent[:, :2] = latent[:, :2] @ rot.T
        latent[:, 2] += 0.25 * np.sin(theta + phase)
        latent[:, 3] += 0.20 * np.cos(theta * 1.5 + phase)

        latent += rng.normal(0.0, noise, size=latent.shape).astype(np.float32)

        high = latent @ projection
        high += rng.normal(0.0, noise * 0.5, size=high.shape).astype(np.float32)
        high += np.float32(slice_idx * 0.15)
        slices.append(high.astype(np.float32))

    stacked = np.vstack(slices)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    normalized = [((sl - mean) / std).astype(np.float32) for sl in slices]
    return normalized


def mean_adjacent_identity_gap(embeddings: List[np.ndarray]) -> float:
    if len(embeddings) < 2:
        return 0.0

    total = 0.0
    count = 0
    for idx in range(len(embeddings) - 1):
        left = embeddings[idx]
        right = embeddings[idx + 1]
        n = min(left.shape[0], right.shape[0])
        delta = left[:n] - right[:n]
        dist = np.linalg.norm(delta, axis=1)
        total += float(dist.sum())
        count += int(n)

    if count == 0:
        return 0.0
    return total / count


def parse_max_rss_mb(time_text: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
    if not match:
        return float("nan")
    return float(match.group(1)) / 1024.0


def parse_json_payload(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON payload found in process output")


def run_timed_command(cmd: List[str], env: Dict[str, str]) -> Tuple[subprocess.CompletedProcess, float]:
    with tempfile.NamedTemporaryFile(prefix="aligned-time-", suffix=".txt", delete=False) as fh:
        time_path = Path(fh.name)

    try:
        full_cmd = ["/usr/bin/time", "-v", "-o", str(time_path), *cmd]
        proc = subprocess.run(full_cmd, capture_output=True, text=True, env=env)
        time_text = time_path.read_text(encoding="utf-8")
        rss_mb = parse_max_rss_mb(time_text)
    finally:
        try:
            time_path.unlink(missing_ok=True)
        except OSError:
            pass

    return proc, rss_mb


def split_stats(values: List[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(v) for v in values],
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
    }


def procrustes_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = y - y.mean(axis=0, keepdims=True)

    x_norm = float(np.linalg.norm(x0))
    y_norm = float(np.linalg.norm(y0))
    if x_norm <= 1e-12 or y_norm <= 1e-12:
        return float("nan")

    x0 /= x_norm
    y0 /= y_norm

    u, _, vt = np.linalg.svd(x0.T @ y0, full_matrices=False)
    rot = u @ vt
    aligned = x0 @ rot
    diff = aligned - y0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def pairwise_distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 3:
        return float("nan")

    dx = x[:, None, :] - x[None, :, :]
    dy = y[:, None, :] - y[None, :, :]

    pdx = np.linalg.norm(dx, axis=2)
    pdy = np.linalg.norm(dy, axis=2)

    iu = np.triu_indices(pdx.shape[0], k=1)
    vx = pdx[iu]
    vy = pdy[iu]

    if np.std(vx) <= 1e-12 or np.std(vy) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(vx, vy)[0, 1])


def consistency_metrics(
    rust_embeddings: List[np.ndarray],
    py_embeddings: List[np.ndarray],
    consistency_sample: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    per_slice_rmse: List[float] = []
    per_slice_dist_corr: List[float] = []

    for idx, (rust_slice, py_slice) in enumerate(zip(rust_embeddings, py_embeddings)):
        n = min(rust_slice.shape[0], py_slice.shape[0])
        rust_use = rust_slice[:n]
        py_use = py_slice[:n]

        if n > consistency_sample:
            choice = np.sort(rng.choice(n, size=consistency_sample, replace=False))
            rust_use = rust_use[choice]
            py_use = py_use[choice]

        rmse = procrustes_rmse(rust_use, py_use)
        corr = pairwise_distance_corr(rust_use, py_use)
        per_slice_rmse.append(rmse)
        per_slice_dist_corr.append(corr)

    rust_gap = mean_adjacent_identity_gap(rust_embeddings)
    py_gap = mean_adjacent_identity_gap(py_embeddings)

    return {
        "slice_procrustes_rmse": [float(v) for v in per_slice_rmse],
        "slice_pairwise_distance_corr": [float(v) for v in per_slice_dist_corr],
        "mean_procrustes_rmse": float(np.nanmean(per_slice_rmse)),
        "mean_pairwise_distance_corr": float(np.nanmean(per_slice_dist_corr)),
        "rust_adjacent_identity_gap": float(rust_gap),
        "python_adjacent_identity_gap": float(py_gap),
        "adjacent_gap_abs_diff": float(abs(rust_gap - py_gap)),
    }


def worker_python(args: argparse.Namespace) -> int:
    try:
        import umap
    except ImportError as exc:
        print(json.dumps({"error": f"umap import failed: {exc}"}))
        return 1

    if not args.slice_csv:
        print(json.dumps({"error": "--slice-csv is required in worker mode"}))
        return 1
    if not args.output_dir:
        print(json.dumps({"error": "--output-dir is required in worker mode"}))
        return 1

    slices = [load_csv(Path(path)) for path in args.slice_csv]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relations = [{i: i for i in range(slices[idx].shape[0])} for idx in range(len(slices) - 1)]

    model = umap.AlignedUMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric=args.metric,
        n_epochs=args.n_epochs,
        learning_rate=1.0,
        min_dist=0.1,
        spread=1.0,
        local_connectivity=1.0,
        set_op_mix_ratio=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        alignment_regularisation=args.alignment_regularization,
        alignment_window_size=2,
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
        verbose=False,
    )

    t0 = time.perf_counter()
    embeddings = model.fit_transform(slices, relations=relations)
    fit_time_sec = time.perf_counter() - t0

    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]

    shape_entries: List[List[int]] = []
    for idx, emb in enumerate(embeddings):
        emb_arr = np.asarray(emb, dtype=np.float32)
        save_csv(out_dir / f"slice_{idx}_embedding.csv", emb_arr)
        shape_entries.append([int(emb_arr.shape[0]), int(emb_arr.shape[1])])

    payload = {
        "fit_time_sec": float(fit_time_sec),
        "adjacent_identity_gap": float(mean_adjacent_identity_gap([np.asarray(e, dtype=np.float32) for e in embeddings])),
        "n_slices": len(embeddings),
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "alignment_regularization": args.alignment_regularization,
        "slice_shapes": shape_entries,
    }
    print(json.dumps(payload))
    return 0


def compare_mode(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.update(THREAD_ENV)

    with tempfile.TemporaryDirectory(prefix="aligned-compare-") as tmpdir:
        tmp = Path(tmpdir)
        data_dir = tmp / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        slices = synthetic_temporal_slices(
            n_slices=args.slices,
            n_samples=args.samples,
            n_features=args.features,
            drift=args.drift,
            noise=args.noise,
            seed=args.seed,
        )

        slice_paths: List[Path] = []
        for idx, sl in enumerate(slices):
            path = data_dir / f"slice_{idx}.csv"
            save_csv(path, sl)
            slice_paths.append(path)

        rust_out = tmp / "rust_out"
        py_out = tmp / "py_out"
        rust_out.mkdir(parents=True, exist_ok=True)
        py_out.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--quiet",
                "--example",
                "aligned_benchmark",
                "--manifest-path",
                str(CRATE_DIR / "Cargo.toml"),
            ],
            check=True,
            env=env,
        )

        python_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-python",
            "--output-dir",
            str(py_out),
            "--n-neighbors",
            str(args.n_neighbors),
            "--n-components",
            str(args.n_components),
            "--n-epochs",
            str(args.n_epochs),
            "--metric",
            args.metric,
            "--seed",
            str(args.seed),
            "--alignment-regularization",
            str(args.alignment_regularization),
            "--alignment-learning-rate",
            str(args.alignment_learning_rate),
            "--alignment-epochs",
            str(args.alignment_epochs),
            "--recenter-interval",
            str(args.recenter_interval),
        ]
        for path in slice_paths:
            python_cmd.extend(["--slice-csv", str(path)])

        rust_cmd = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--example",
            "aligned_benchmark",
            "--manifest-path",
            str(CRATE_DIR / "Cargo.toml"),
            "--",
            str(rust_out),
            *[str(path) for path in slice_paths],
            "--n-neighbors",
            str(args.n_neighbors),
            "--n-components",
            str(args.n_components),
            "--n-epochs",
            str(args.n_epochs),
            "--metric",
            args.metric,
            "--seed",
            str(args.seed),
            "--init",
            "random",
            "--use-approximate-knn",
            "false",
            "--alignment-regularization",
            str(args.alignment_regularization),
            "--alignment-learning-rate",
            str(args.alignment_learning_rate),
            "--alignment-epochs",
            str(args.alignment_epochs),
            "--recenter-interval",
            str(args.recenter_interval),
        ]

        def run_impl(cmd: List[str], impl_name: str) -> Dict[str, object]:
            timing: List[float] = []
            rss_list: List[float] = []
            last_payload: Dict[str, object] = {}

            total_runs = args.warmup + args.repeats
            for run_idx in range(total_runs):
                proc, rss = run_timed_command(cmd, env)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{impl_name} run failed (run {run_idx + 1}/{total_runs}):\n"
                        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                    )

                payload = parse_json_payload(proc.stdout)
                if run_idx >= args.warmup:
                    timing.append(float(payload["fit_time_sec"]))
                    rss_list.append(float(rss))
                    last_payload = payload

            return {
                "fit_time_sec": split_stats(timing),
                "max_rss_mb": split_stats(rss_list),
                "last_payload": last_payload,
            }

        py_stats = run_impl(python_cmd, "python_aligned_umap")
        rust_stats = run_impl(rust_cmd, "rust_aligned_umap")

        py_embeddings = [load_csv(py_out / f"slice_{idx}_embedding.csv") for idx in range(args.slices)]
        rust_embeddings = [load_csv(rust_out / f"slice_{idx}_embedding.csv") for idx in range(args.slices)]

        consistency = consistency_metrics(
            rust_embeddings=rust_embeddings,
            py_embeddings=py_embeddings,
            consistency_sample=args.consistency_sample,
            seed=args.seed,
        )

        speed_ratio = float(rust_stats["fit_time_sec"]["mean"] / py_stats["fit_time_sec"]["mean"])
        memory_ratio = float(rust_stats["max_rss_mb"]["mean"] / py_stats["max_rss_mb"]["mean"])

        report: Dict[str, object] = {
            "config": {
                "slices": args.slices,
                "samples": args.samples,
                "features": args.features,
                "drift": args.drift,
                "noise": args.noise,
                "metric": args.metric,
                "n_neighbors": args.n_neighbors,
                "n_components": args.n_components,
                "n_epochs": args.n_epochs,
                "seed": args.seed,
                "alignment_regularization": args.alignment_regularization,
                "alignment_learning_rate": args.alignment_learning_rate,
                "alignment_epochs": args.alignment_epochs,
                "recenter_interval": args.recenter_interval,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "consistency_sample": args.consistency_sample,
            },
            "bias_controls": {
                "shared_synthetic_data": True,
                "global_feature_standardization": True,
                "same_seed": args.seed,
                "metric_alignment": args.metric,
                "single_thread_env": THREAD_ENV,
                "identity_relations": True,
                "same_neighbors_components_epochs": True,
            },
            "python_aligned_umap": py_stats,
            "rust_aligned_umap": rust_stats,
            "consistency": consistency,
            "summary": {
                "speed_ratio_rust_over_python": speed_ratio,
                "memory_ratio_rust_over_python": memory_ratio,
                "mean_procrustes_rmse": consistency["mean_procrustes_rmse"],
                "mean_pairwise_distance_corr": consistency["mean_pairwise_distance_corr"],
                "adjacent_gap_abs_diff": consistency["adjacent_gap_abs_diff"],
            },
        }

        text = json.dumps(report, indent=2)
        print(text)

        if args.output_json:
            Path(args.output_json).write_text(text + "\n", encoding="utf-8")

    return 0


def main() -> int:
    args = parse_args()
    if args.worker_python:
        return worker_python(args)
    return compare_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
