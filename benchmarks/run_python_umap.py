#!/usr/bin/env python3
import argparse
import numpy as np
import umap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Python umap-learn on CSV data")
    p.add_argument("--input", required=True, help="input CSV path")
    p.add_argument("--output", required=True, help="output embedding CSV path")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init", choices=["random", "spectral"], default="random")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    model = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric="euclidean",
        n_epochs=args.n_epochs,
        learning_rate=1.0,
        init=args.init,
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
        verbose=False,
    )

    embedding = model.fit_transform(x).astype(np.float32)
    np.savetxt(args.output, embedding, delimiter=",", fmt="%.8f")


if __name__ == "__main__":
    main()
