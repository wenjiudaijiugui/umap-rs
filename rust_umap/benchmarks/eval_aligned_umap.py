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
    parser.add_argument("--init", choices=["random", "spectral"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--knn-strategy",
        choices=["exact", "approximate", "auto"],
        default="exact",
        help="exact: force both sides exact; approximate: force both sides approximate; auto: implementation defaults with strict threshold guard",
    )
    parser.add_argument(
        "--approx-knn-threshold",
        type=int,
        default=4096,
        help="shared threshold used by Rust auto strategy; in auto mode must stay 4096 to match umap-learn behavior",
    )
    parser.add_argument("--approx-knn-candidates", type=int, default=30)
    parser.add_argument("--approx-knn-iters", type=int, default=10)

    parser.add_argument("--alignment-regularization", type=float, default=0.08)
    parser.add_argument("--alignment-learning-rate", type=float, default=0.25)
    parser.add_argument("--alignment-epochs", type=int, default=100)
    parser.add_argument("--recenter-interval", type=int, default=5)

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consistency-sample", type=int, default=256)
    parser.add_argument("--gate-max-mean-procrustes-rmse", type=float, default=0.12)
    parser.add_argument("--gate-min-mean-pairwise-distance-corr", type=float, default=0.70)
    parser.add_argument("--gate-max-adjacent-gap-abs-diff", type=float, default=2.50)
    parser.add_argument("--gate-max-speed-ratio", type=float, default=3.0)
    parser.add_argument("--gate-max-memory-ratio", type=float, default=3.0)
    parser.add_argument("--no-fail-on-gate", action="store_true")
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


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def resolve_knn_strategy(
    strategy: str,
    approx_knn_threshold: int,
    max_samples_per_slice: int,
) -> Dict[str, object]:
    if strategy == "exact":
        # umap-learn falls back to ANN for large inputs unless approximation is forced off
        # and sample count stays below its exact-distance path threshold.
        exact_path_guaranteed = max_samples_per_slice <= approx_knn_threshold
        return {
            "mode": "exact",
            "rust_use_approximate_knn": False,
            "rust_approx_knn_threshold": approx_knn_threshold,
            "python_force_approximation_algorithm": False,
            "python_exact_path_guaranteed": bool(exact_path_guaranteed),
            "constraint_pass": bool(exact_path_guaranteed),
            "constraint_reason": (
                "exact mode requires max_samples_per_slice <= approx_knn_threshold "
                "to prevent umap-learn from switching to approximate neighbors"
            ),
        }

    if strategy == "approximate":
        return {
            "mode": "approximate",
            "rust_use_approximate_knn": True,
            "rust_approx_knn_threshold": 0,
            "python_force_approximation_algorithm": True,
            "python_exact_path_guaranteed": False,
            "constraint_pass": True,
            "constraint_reason": "both implementations are forced to ANN neighbor search",
        }

    # umap-learn's auto mode uses a fixed internal threshold (~4096). Keep Rust aligned.
    auto_threshold_ok = approx_knn_threshold == 4096
    return {
        "mode": "auto",
        "rust_use_approximate_knn": True,
        "rust_approx_knn_threshold": approx_knn_threshold,
        "python_force_approximation_algorithm": False,
        "python_exact_path_guaranteed": max_samples_per_slice <= 4096,
        "constraint_pass": bool(auto_threshold_ok),
        "constraint_reason": (
            "auto mode parity requires approx_knn_threshold == 4096 to mirror "
            "umap-learn's internal exact/approx switch"
        ),
    }


def paired_alternating_schedule(
    impl_names: List[str],
    rounds: int,
    seed: int,
) -> Dict[str, object]:
    if len(impl_names) != 2:
        raise ValueError("paired alternating schedule expects exactly two implementations")
    rng = np.random.default_rng(seed ^ 0xA11CED)
    first_idx = int(rng.integers(0, 2))
    first = impl_names[first_idx]
    second = impl_names[1 - first_idx]

    orders: List[List[str]] = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            orders.append([first, second])
        else:
            orders.append([second, first])

    return {
        "strategy": "paired_alternating_random_start",
        "random_start_first_impl": first,
        "round_orders": orders,
    }


def compare_leq(name: str, value: float, threshold: float) -> Dict[str, object]:
    passed = bool(np.isfinite(value) and value <= threshold)
    return {
        "metric": name,
        "value": float(value),
        "operator": "<=",
        "threshold": float(threshold),
        "pass": passed,
    }


def compare_geq(name: str, value: float, threshold: float) -> Dict[str, object]:
    passed = bool(np.isfinite(value) and value >= threshold)
    return {
        "metric": name,
        "value": float(value),
        "operator": ">=",
        "threshold": float(threshold),
        "pass": passed,
    }


def build_gate_results(
    args: argparse.Namespace,
    consistency: Dict[str, object],
    speed_ratio: float,
    memory_ratio: float,
) -> Dict[str, object]:
    consistency_checks = [
        compare_leq(
            "mean_procrustes_rmse",
            float(consistency["mean_procrustes_rmse"]),
            float(args.gate_max_mean_procrustes_rmse),
        ),
        compare_geq(
            "mean_pairwise_distance_corr",
            float(consistency["mean_pairwise_distance_corr"]),
            float(args.gate_min_mean_pairwise_distance_corr),
        ),
        compare_leq(
            "adjacent_gap_abs_diff",
            float(consistency["adjacent_gap_abs_diff"]),
            float(args.gate_max_adjacent_gap_abs_diff),
        ),
    ]
    consistency_pass = all(check["pass"] for check in consistency_checks)

    speed_checks = [
        compare_leq(
            "speed_ratio_rust_over_python",
            speed_ratio,
            float(args.gate_max_speed_ratio),
        )
    ]
    speed_pass = all(check["pass"] for check in speed_checks)

    memory_checks = [
        compare_leq(
            "memory_ratio_rust_over_python",
            memory_ratio,
            float(args.gate_max_memory_ratio),
        )
    ]
    memory_pass = all(check["pass"] for check in memory_checks)

    failing_sections: List[str] = []
    if not consistency_pass:
        failing_sections.append("consistency")
    if not speed_pass:
        failing_sections.append("speed")
    if not memory_pass:
        failing_sections.append("memory")

    return {
        "consistency": {
            "pass": consistency_pass,
            "checks": consistency_checks,
            "threshold_reason": (
                "Use strict shape agreement (Procrustes RMSE), moderate pairwise-structure agreement "
                "(distance correlation), and a coarse adjacent-gap alarm. The gap threshold is looser "
                "because it is sensitive to implementation-specific global scale."
            ),
        },
        "speed": {
            "pass": speed_pass,
            "checks": speed_checks,
            "threshold_reason": (
                "Prevent severe runtime regressions while allowing moderate implementation variance."
            ),
        },
        "memory": {
            "pass": memory_pass,
            "checks": memory_checks,
            "threshold_reason": (
                "Prevent severe peak RSS regressions while tolerating allocator/runtime differences."
            ),
        },
        "overall": {
            "pass": not failing_sections,
            "failing_sections": failing_sections,
        },
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
    max_samples_per_slice = max(int(sl.shape[0]) for sl in slices)

    knn_cfg = resolve_knn_strategy(
        strategy=args.knn_strategy,
        approx_knn_threshold=args.approx_knn_threshold,
        max_samples_per_slice=max_samples_per_slice,
    )
    if not bool(knn_cfg["constraint_pass"]):
        print(
            json.dumps(
                {
                    "error": "knn strategy constraint failed",
                    "knn_strategy": knn_cfg,
                }
            )
        )
        return 1

    relations = [{i: i for i in range(slices[idx].shape[0])} for idx in range(len(slices) - 1)]

    model = umap.AlignedUMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric=args.metric,
        init=args.init,
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
        force_approximation_algorithm=bool(knn_cfg["python_force_approximation_algorithm"]),
        low_memory=False,
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
        "init": args.init,
        "knn_strategy": knn_cfg["mode"],
        "python_force_approximation_algorithm": knn_cfg["python_force_approximation_algorithm"],
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

        max_samples_per_slice = max(int(sl.shape[0]) for sl in slices)
        knn_cfg = resolve_knn_strategy(
            strategy=args.knn_strategy,
            approx_knn_threshold=args.approx_knn_threshold,
            max_samples_per_slice=max_samples_per_slice,
        )
        if not bool(knn_cfg["constraint_pass"]):
            raise RuntimeError(
                f"knn strategy constraint failed: {knn_cfg['constraint_reason']} "
                f"(mode={knn_cfg['mode']}, max_samples_per_slice={max_samples_per_slice}, "
                f"approx_knn_threshold={args.approx_knn_threshold})"
            )

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
            "--init",
            args.init,
            "--knn-strategy",
            args.knn_strategy,
            "--approx-knn-threshold",
            str(args.approx_knn_threshold),
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
            args.init,
            "--use-approximate-knn",
            str(bool(knn_cfg["rust_use_approximate_knn"])).lower(),
            "--approx-knn-candidates",
            str(args.approx_knn_candidates),
            "--approx-knn-iters",
            str(args.approx_knn_iters),
            "--approx-knn-threshold",
            str(knn_cfg["rust_approx_knn_threshold"]),
            "--alignment-regularization",
            str(args.alignment_regularization),
            "--alignment-learning-rate",
            str(args.alignment_learning_rate),
            "--alignment-epochs",
            str(args.alignment_epochs),
            "--recenter-interval",
            str(args.recenter_interval),
        ]

        impl_cmds: Dict[str, List[str]] = {
            "python_aligned_umap": python_cmd,
            "rust_aligned_umap": rust_cmd,
        }

        schedule_meta = paired_alternating_schedule(
            impl_names=list(impl_cmds.keys()),
            rounds=args.warmup + args.repeats,
            seed=args.seed,
        )
        schedule = schedule_meta["round_orders"]

        raw_stats: Dict[str, Dict[str, object]] = {
            impl_name: {"timing": [], "rss": [], "last_payload": {}}
            for impl_name in impl_cmds.keys()
        }

        for round_idx, order in enumerate(schedule):
            measured = round_idx >= args.warmup
            for impl_name in order:
                proc, rss = run_timed_command(impl_cmds[impl_name], env)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{impl_name} run failed (round {round_idx + 1}/{len(schedule)}):\n"
                        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
                    )

                payload = parse_json_payload(proc.stdout)
                if "error" in payload:
                    raise RuntimeError(
                        f"{impl_name} returned error payload (round {round_idx + 1}/{len(schedule)}): "
                        f"{payload['error']}"
                    )

                if measured:
                    raw_stats[impl_name]["timing"].append(float(payload["fit_time_sec"]))
                    raw_stats[impl_name]["rss"].append(float(rss))
                    raw_stats[impl_name]["last_payload"] = payload

        py_stats = {
            "fit_time_sec": split_stats(raw_stats["python_aligned_umap"]["timing"]),
            "max_rss_mb": split_stats(raw_stats["python_aligned_umap"]["rss"]),
            "last_payload": raw_stats["python_aligned_umap"]["last_payload"],
        }
        rust_stats = {
            "fit_time_sec": split_stats(raw_stats["rust_aligned_umap"]["timing"]),
            "max_rss_mb": split_stats(raw_stats["rust_aligned_umap"]["rss"]),
            "last_payload": raw_stats["rust_aligned_umap"]["last_payload"],
        }

        py_embeddings = [load_csv(py_out / f"slice_{idx}_embedding.csv") for idx in range(args.slices)]
        rust_embeddings = [load_csv(rust_out / f"slice_{idx}_embedding.csv") for idx in range(args.slices)]

        consistency = consistency_metrics(
            rust_embeddings=rust_embeddings,
            py_embeddings=py_embeddings,
            consistency_sample=args.consistency_sample,
            seed=args.seed,
        )

        speed_ratio = safe_ratio(
            float(rust_stats["fit_time_sec"]["mean"]),
            float(py_stats["fit_time_sec"]["mean"]),
        )
        memory_ratio = safe_ratio(
            float(rust_stats["max_rss_mb"]["mean"]),
            float(py_stats["max_rss_mb"]["mean"]),
        )
        gates = build_gate_results(
            args=args,
            consistency=consistency,
            speed_ratio=speed_ratio,
            memory_ratio=memory_ratio,
        )

        report: Dict[str, object] = {
            "config": {
                "slices": args.slices,
                "samples": args.samples,
                "features": args.features,
                "drift": args.drift,
                "noise": args.noise,
                "metric": args.metric,
                "init": args.init,
                "knn_strategy": args.knn_strategy,
                "approx_knn_threshold": args.approx_knn_threshold,
                "approx_knn_candidates": args.approx_knn_candidates,
                "approx_knn_iters": args.approx_knn_iters,
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
                "gate_max_mean_procrustes_rmse": args.gate_max_mean_procrustes_rmse,
                "gate_min_mean_pairwise_distance_corr": args.gate_min_mean_pairwise_distance_corr,
                "gate_max_adjacent_gap_abs_diff": args.gate_max_adjacent_gap_abs_diff,
                "gate_max_speed_ratio": args.gate_max_speed_ratio,
                "gate_max_memory_ratio": args.gate_max_memory_ratio,
            },
            "bias_controls": {
                "shared_synthetic_data": True,
                "global_feature_standardization": True,
                "same_seed": args.seed,
                "metric_alignment": args.metric,
                "init_alignment": args.init,
                "single_thread_env": THREAD_ENV,
                "identity_relations": True,
                "same_neighbors_components_epochs": True,
                "knn_strategy_alignment": knn_cfg,
                "run_order_debias": schedule_meta,
                "python_low_memory_disabled_for_parity": True,
                "python_n_jobs_flag_available_in_aligned_umap": False,
                "rust_specific_knn_params_reported": True,
            },
            "python_aligned_umap": py_stats,
            "rust_aligned_umap": rust_stats,
            "consistency": consistency,
            "gates": gates,
            "summary": {
                "speed_ratio_rust_over_python": speed_ratio,
                "memory_ratio_rust_over_python": memory_ratio,
                "mean_procrustes_rmse": consistency["mean_procrustes_rmse"],
                "mean_pairwise_distance_corr": consistency["mean_pairwise_distance_corr"],
                "adjacent_gap_abs_diff": consistency["adjacent_gap_abs_diff"],
                "hard_gate_pass": bool(gates["overall"]["pass"]),
            },
        }

        text = json.dumps(report, indent=2)
        print(text)

        if args.output_json:
            Path(args.output_json).write_text(text + "\n", encoding="utf-8")

    if not bool(gates["overall"]["pass"]) and not args.no_fail_on_gate:
        for section in gates["overall"]["failing_sections"]:
            print(f"FAIL: aligned hard gate failed at section '{section}'", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_python:
        return worker_python(args)
    return compare_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
