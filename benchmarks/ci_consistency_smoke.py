#!/usr/bin/env python3
import argparse
import json
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
    "breast_cancer": lambda: StandardScaler().fit_transform(load_breast_cancer().data.astype(np.float32)).astype(np.float32),
    "digits": lambda: StandardScaler().fit_transform(load_digits().data.astype(np.float32)).astype(np.float32),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI smoke consistency gate against public UMAP implementations")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--rscript-bin", default=shutil.which("Rscript") or "")
    p.add_argument("--require-r", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--sample-cap", type=int, default=1500)
    p.add_argument("--trust-gap", type=float, default=0.02)
    p.add_argument("--recall-gap", type=float, default=0.05)
    p.add_argument("--pairwise-min-overlap", type=float, default=0.40)
    return p.parse_args()


def build_release_bins() -> None:
    fair.build_rust_binaries()


def available_impls(args: argparse.Namespace) -> List[str]:
    impls = ["python_umap_learn", "rust_umap"]
    has_r = bool(args.rscript_bin) and shutil.which(args.rscript_bin) is not None if Path(args.rscript_bin).name == args.rscript_bin else Path(args.rscript_bin).exists()
    if args.require_r and not has_r:
        raise RuntimeError("--require-r was set but Rscript is not available")
    if has_r:
        impls.insert(1, "r_uwot")
    return impls


def summarize_dataset(name: str, consistency: Dict[str, object], impls: List[str], trust_gap: float, recall_gap: float, min_overlap: float) -> List[str]:
    failures: List[str] = []
    trust = consistency["trustworthiness_at_15"]
    recall = consistency["original_knn_recall_at_15"]
    pairwise = consistency["pairwise"]

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
        if overlap < min_overlap:
            failures.append(
                f"{name}: pairwise overlap rust vs {impl} fell to {overlap:.6f}, below minimum {min_overlap:.6f}"
            )
    return failures


def main() -> None:
    args = parse_args()
    fair.PYTHON_BIN = Path(args.python_bin)
    fair.RSCRIPT_BIN = Path(args.rscript_bin) if args.rscript_bin else Path("Rscript")
    build_release_bins()
    impls = available_impls(args)

    all_failures: List[str] = []
    summary: Dict[str, object] = {"impls": impls, "datasets": {}}

    with tempfile.TemporaryDirectory(prefix="umap-rs-ci-consistency-") as tmpdir:
        tmp = Path(tmpdir)
        fair.DATA_DIR = tmp / "data"
        fair.KNN_DIR = tmp / "knn"
        fair.OUT_DIR = tmp / "out"
        fair.TIME_DIR = tmp / "time"
        fair.ensure_dirs()

        for name, loader in DATASETS.items():
            x = loader()
            data_path = fair.DATA_DIR / f"{name}.csv"
            idx_path = fair.KNN_DIR / f"{name}_idx.csv"
            dist_path = fair.KNN_DIR / f"{name}_dist.csv"
            fair.save_dataset_csv(data_path, x)
            orig_knn_idx = fair.compute_shared_exact_knn(x, fair.N_NEIGHBORS, idx_path, dist_path)

            embeddings = {}
            run_meta = {}
            for impl in impls:
                out_path = fair.OUT_DIR / f"{name}__{impl}.csv"
                time_path = fair.TIME_DIR / f"{name}__{impl}.time.txt"
                cmd = fair.algo_exact_cmd(
                    impl,
                    data_path,
                    out_path,
                    args.seed,
                    args.warmup,
                    args.repeats,
                    idx_path,
                    dist_path,
                )
                run = fair.run_timed(cmd, time_path)
                payload = fair.parse_json_line(run.stdout)
                emb = np.loadtxt(out_path, delimiter=",", dtype=np.float32)
                if emb.ndim == 1:
                    emb = emb.reshape(-1, fair.N_COMPONENTS)
                embeddings[impl] = emb
                run_meta[impl] = {
                    "fit_mean_sec": float(payload.get("fit_mean_sec", float("nan"))),
                    "fit_std_sec": float(payload.get("fit_std_sec", float("nan"))),
                    "process_max_rss_mb": float(run.max_rss_mb),
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
