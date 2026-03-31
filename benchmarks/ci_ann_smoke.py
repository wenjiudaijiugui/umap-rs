#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import compare_real_impls_fair as fair

DATASETS = {
    "breast_cancer": lambda: StandardScaler()
    .fit_transform(load_breast_cancer().data.astype(np.float32))
    .astype(np.float32),
    "digits": lambda: StandardScaler()
    .fit_transform(load_digits().data.astype(np.float32))
    .astype(np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CI ANN/e2e smoke gate against public UMAP implementation"
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--rscript-bin", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-cap", type=int, default=1200)
    parser.add_argument("--trust-gap", type=float, default=0.03)
    parser.add_argument("--recall-gap", type=float, default=0.08)
    parser.add_argument("--pairwise-min-overlap", type=float, default=0.35)
    return parser.parse_args()


def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric detected for {name}: {value}")


def _check_inputs(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.sample_cap < 1:
        raise ValueError("--sample-cap must be >= 1")
    if not (0.0 <= args.pairwise_min_overlap <= 1.0):
        raise ValueError("--pairwise-min-overlap must be in [0, 1]")


def summarize_dataset(
    name: str,
    consistency: Dict[str, object],
    impls: List[str],
    trust_gap: float,
    recall_gap: float,
    min_overlap: float,
) -> List[str]:
    failures: List[str] = []
    trust = consistency["trustworthiness_at_15"]
    recall = consistency["original_knn_recall_at_15"]
    pairwise = consistency["pairwise"]

    for impl in impls:
        _ensure_finite(f"{name}.trustworthiness[{impl}]", float(trust[impl]))
        _ensure_finite(f"{name}.recall[{impl}]", float(recall[impl]))

    public_impls = [impl for impl in impls if impl != "rust_umap"]
    rust_trust = float(trust["rust_umap"])
    rust_recall = float(recall["rust_umap"])
    public_best_trust = max(float(trust[impl]) for impl in public_impls)
    public_best_recall = max(float(recall[impl]) for impl in public_impls)

    if rust_trust + trust_gap < public_best_trust:
        failures.append(
            f"{name}: rust trustworthiness {rust_trust:.6f} is worse than public best {public_best_trust:.6f} by more than {trust_gap:.6f}"
        )
    if rust_recall + recall_gap < public_best_recall:
        failures.append(
            f"{name}: rust original_knn_recall {rust_recall:.6f} is worse than public best {public_best_recall:.6f} by more than {recall_gap:.6f}"
        )

    for impl in public_impls:
        key = "__vs__".join(sorted([impl, "rust_umap"]))
        overlap = float(pairwise[key]["knn_overlap_at_15"])
        _ensure_finite(f"{name}.pairwise_overlap[{impl}]", overlap)
        if overlap < min_overlap:
            failures.append(
                f"{name}: pairwise overlap rust vs {impl} fell to {overlap:.6f}, below minimum {min_overlap:.6f}"
            )

    return failures


def main() -> None:
    args = parse_args()
    _check_inputs(args)

    fair.PYTHON_BIN = Path(args.python_bin)
    fair.RSCRIPT_BIN = Path(args.rscript_bin) if args.rscript_bin else Path("Rscript")

    # Explicitly keep this gate lightweight and deterministic.
    impls = ["python_umap_learn", "rust_umap"]

    fair.build_rust_binaries()

    all_failures: List[str] = []
    summary: Dict[str, object] = {
        "impls": impls,
        "datasets": {},
        "config": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "sample_cap": args.sample_cap,
            "thread_env": fair.THREAD_ENV,
            "timing_mode": "e2e_default_ann",
        },
    }

    with tempfile.TemporaryDirectory(prefix="umap-rs-ci-ann-e2e-") as tmpdir:
        tmp = Path(tmpdir)
        fair.DATA_DIR = tmp / "data"
        fair.KNN_DIR = tmp / "knn"
        fair.OUT_DIR = tmp / "out"
        fair.TIME_DIR = tmp / "time"
        fair.ensure_dirs()

        for name, loader in DATASETS.items():
            x = loader()
            data_path = fair.DATA_DIR / f"{name}.csv"
            fair.save_dataset_csv(data_path, x)

            idx_path = fair.KNN_DIR / f"{name}_idx.csv"
            dist_path = fair.KNN_DIR / f"{name}_dist.csv"
            orig_knn_idx = fair.compute_shared_exact_knn(
                x, fair.N_NEIGHBORS, idx_path, dist_path
            )

            embeddings: Dict[str, np.ndarray] = {}
            run_meta: Dict[str, object] = {}

            for impl in impls:
                out_path = fair.OUT_DIR / f"{name}__{impl}.csv"
                time_path = fair.TIME_DIR / f"{name}__{impl}.time.txt"

                elapsed_samples = []
                rss_samples = []
                total = args.warmup + args.repeats
                for rep in range(total):
                    cmd = fair.e2e_cmd(impl, data_path, out_path, args.seed + rep)
                    run = fair.run_timed(cmd, time_path)
                    if rep >= args.warmup:
                        _ensure_finite(
                            f"{name}.{impl}.elapsed_sec_rep{rep}", float(run.elapsed_sec)
                        )
                        _ensure_finite(
                            f"{name}.{impl}.max_rss_mb_rep{rep}", float(run.max_rss_mb)
                        )
                        elapsed_samples.append(float(run.elapsed_sec))
                        rss_samples.append(float(run.max_rss_mb))

                emb = np.loadtxt(out_path, delimiter=",", dtype=np.float32)
                if emb.ndim == 1:
                    emb = emb.reshape(-1, fair.N_COMPONENTS)
                embeddings[impl] = emb

                run_meta[impl] = {
                    "elapsed_mean_sec": float(np.mean(elapsed_samples)),
                    "elapsed_std_sec": float(np.std(elapsed_samples)),
                    "max_rss_mean_mb": float(np.mean(rss_samples)),
                    "max_rss_std_mb": float(np.std(rss_samples)),
                    "elapsed_samples_sec": elapsed_samples,
                    "max_rss_samples_mb": rss_samples,
                }

            consistency = fair.compute_consistency(
                x=x,
                embeddings=embeddings,
                seed=args.seed,
                orig_knn_idx=orig_knn_idx,
                k=fair.N_NEIGHBORS,
                sample_cap=args.sample_cap,
            )

            summary["datasets"][name] = {
                "runs": run_meta,
                "consistency": consistency,
            }

            all_failures.extend(
                summarize_dataset(
                    name,
                    consistency,
                    impls,
                    args.trust_gap,
                    args.recall_gap,
                    args.pairwise_min_overlap,
                )
            )

    print(json.dumps(summary, indent=2))
    if all_failures:
        for failure in all_failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
