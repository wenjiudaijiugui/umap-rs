#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np

try:
    from rust_umap_py import Umap
except Exception as exc:  # pragma: no cover - environment wiring
    raise SystemExit(
        "failed to import rust_umap_py; run `maturin develop --manifest-path rust_umap_py/Cargo.toml` first"
    ) from exc


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rust_umap_py in algorithm-only timing mode")
    p.add_argument("--input", required=True, help="input CSV path")
    p.add_argument("--output", required=True, help="output embedding CSV path")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    p.add_argument("--metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--knn-indices", default="", help="optional precomputed kNN indices CSV")
    p.add_argument("--knn-dists", default="", help="optional precomputed kNN distances CSV")
    p.add_argument("--knn-metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    p.add_argument("--use-approximate-knn", type=parse_bool, default=False)
    p.add_argument("--approx-knn-candidates", type=int, default=30)
    p.add_argument("--approx-knn-iters", type=int, default=10)
    p.add_argument("--approx-knn-threshold", type=int, default=4096)
    return p.parse_args()


def load_precomputed_knn(args: argparse.Namespace):
    if not args.knn_indices and not args.knn_dists:
        return None
    if not args.knn_indices or not args.knn_dists:
        raise ValueError("both --knn-indices and --knn-dists must be provided")

    knn_idx = np.loadtxt(args.knn_indices, delimiter=",", dtype=np.int64)
    knn_dist = np.loadtxt(args.knn_dists, delimiter=",", dtype=np.float32)
    if knn_idx.ndim == 1:
        knn_idx = knn_idx.reshape(1, -1)
    if knn_dist.ndim == 1:
        knn_dist = knn_dist.reshape(1, -1)
    if knn_idx.shape != knn_dist.shape:
        raise ValueError("precomputed knn indices and distances must have identical shapes")

    return knn_idx, knn_dist


def create_model(args: argparse.Namespace) -> Umap:
    return Umap(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric=args.metric,
        n_epochs=args.n_epochs,
        learning_rate=1.0,
        init=args.init,
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        random_seed=args.seed,
        use_approximate_knn=args.use_approximate_knn,
        approx_knn_candidates=args.approx_knn_candidates,
        approx_knn_iters=args.approx_knn_iters,
        approx_knn_threshold=args.approx_knn_threshold,
    )


def main() -> None:
    args = parse_args()
    x = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    precomputed_knn = load_precomputed_knn(args)
    embedding = np.empty((x.shape[0], args.n_components), dtype=np.float32)

    fit_times = []
    total = args.warmup + args.repeats
    for i in range(total):
        model = create_model(args)
        t0 = time.perf_counter()
        if precomputed_knn is None:
            model.fit_transform(x, out=embedding)
        else:
            knn_idx, knn_dist = precomputed_knn
            model.fit_transform_with_knn(
                x,
                knn_idx,
                knn_dist,
                knn_metric=args.knn_metric,
                out=embedding,
            )
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            fit_times.append(dt)

    np.savetxt(args.output, embedding, delimiter=",", fmt="%.8f")

    fit_arr = np.asarray(fit_times, dtype=np.float64)
    result = {
        "mode": "fit",
        "metric": args.metric,
        "knn_metric": args.knn_metric,
        "precomputed_knn": precomputed_knn is not None,
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "fit_times_sec": fit_times,
        "fit_mean_sec": float(np.mean(fit_arr)) if fit_arr.size else 0.0,
        "fit_std_sec": float(np.std(fit_arr)) if fit_arr.size else 0.0,
        "interop": {
            "input_dtype": str(x.dtype),
            "input_c_contiguous": bool(x.flags.c_contiguous),
            "embedding_buffer_reused": True,
        },
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
