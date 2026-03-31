# rust_umap

A ground-up Rust implementation of the core UMAP pipeline for reproducible algorithm study.

## What is implemented

- KNN search with multiple metrics:
  - Exact all-pairs baseline for `euclidean`, `manhattan`/`l1`, and `cosine`
  - Optional NN-Descent-style approximate mode for larger datasets
- Smooth KNN distance estimation (`sigma`, `rho`) with binary search
- Fuzzy simplicial set construction and symmetric fuzzy union
- Automatic `(a, b)` curve parameter fitting from `(spread, min_dist)`
- Two initialization strategies:
  - `InitMethod::Spectral` (normalized graph Laplacian eigenvectors)
  - `InitMethod::Random`
- Spectral initialization for disconnected graphs via component-aware layout
- Stochastic layout optimization with negative sampling
- End-to-end training APIs:
  - `fit_transform`
  - `fit`
- New point embedding API:
  - `transform`
- Sparse CSR input MVP:
  - `SparseCsrMatrix` + `fit_sparse_csr` / `fit_transform_sparse_csr`
  - Exact Euclidean sparse kNN (no full dense distance matrix materialization)
  - `fit_csv` / `bench_fit_csv` support `--csr-indptr/--csr-indices/--csr-data/--csr-n-cols`
  - Sparse `inverse_transform` is intentionally unsupported in this MVP
- Approximate inverse mapping API:
  - `inverse_transform` (Euclidean metric in embedding space)
  - Exact training-embedding lookups are mapped back to their original samples
- CLI utilities:
  - `fit_csv`
  - `bench_fit_csv`
  - optional `--metric euclidean|manhattan|cosine`
  - optional `--knn-metric ...` guard in precomputed-kNN mode

## What is intentionally out of scope (for this version)

- Full `pynndescent`-equivalent ANN quality/performance parity
- Sparse path parity for non-Euclidean metrics
- Sparse-trained inverse transform

## Quick start

```bash
cd rust_umap
cargo test
cargo run --release
```

`cargo run` executes a toy dataset embedding and then transforms a small query subset.

## CI-facing validation

Repository automation keeps benchmark validation staged in this order:

1. Compare consistency against public implementations.
2. Check ANN/e2e smoke behavior against public implementation baselines.
3. Check for no-regression on timing and memory (metric matrix: euclidean/manhattan/cosine).
4. Run a deeper optimization-stage benchmark report when explicitly requested.

The Rust crate remains the unit under test in all staged checks.

For CSV-driven runs:

```bash
cargo run --release --bin fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 fit \
  --metric cosine
```

For sparse CSR input (`fit` mode), pass CSR arrays directly:

```bash
cargo run --release --bin fit_csv -- \
  dummy.csv embedding.csv \
  15 2 200 42 random false 30 10 4096 fit \
  --metric euclidean \
  --csr-indptr indptr.csv \
  --csr-indices indices.csv \
  --csr-data values.csv \
  --csr-n-cols 1800
```

For precomputed kNN input, add explicit metric guard:

```bash
cargo run --release --bin fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 fit_precomputed \
  knn_idx.csv knn_dist.csv \
  --metric cosine \
  --knn-metric cosine
```

To benchmark repeated fits:

```bash
cargo run --release --bin bench_fit_csv -- \
  data.csv embedding.csv \
  15 2 200 42 spectral false 30 10 4096 1 5 \
  --metric manhattan
```

To benchmark sparse CSR fits against `umap-learn` (consistency + speed + memory):

```bash
uv run --with numpy --with scipy --with scikit-learn --with umap-learn \
  python rust_umap/benchmarks/eval_sparse_csr_vs_umap_learn.py --dataset all
```

## Minimal library usage

```rust
use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};

let data: Vec<Vec<f32>> = vec![
    vec![0.0, 0.1, 0.2],
    vec![0.2, 0.1, 0.0],
    // ...
];

let mut model = UmapModel::new(UmapParams {
    init: InitMethod::Spectral,
    metric: Metric::Cosine,
    ..UmapParams::default()
});

let embedding = model.fit_transform(&data)?;
let new_points = vec![vec![0.1, 0.1, 0.2]];
let transformed = model.transform(&new_points)?;
let reconstructed = model.inverse_transform(&transformed)?;
```

Minimal sparse usage:

```rust
use rust_umap::{fit_transform_sparse_csr, Metric, SparseCsrMatrix, UmapParams};

let csr = SparseCsrMatrix::new(
    3,
    5,
    vec![0, 2, 3, 5],
    vec![0, 3, 1, 0, 4],
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
)?;

let embedding = fit_transform_sparse_csr(
    csr,
    UmapParams {
        metric: Metric::Euclidean,
        ..UmapParams::default()
    },
)?;
```

## Notes on fidelity

The core equations for graph construction and layout optimization are aligned with standard UMAP behavior.
This implementation prioritizes clarity and reproducibility over large-scale performance.

## Inverse benchmark helpers

For inverse-transform quality and Euclidean fit no-regression checks:

```bash
python rust_umap/benchmarks/eval_inverse_quality.py --dataset all
python rust_umap/benchmarks/eval_euclidean_fit_regression.py --dataset all \
  --output-json rust_umap/benchmarks/current_fit_regression.json
python rust_umap/benchmarks/eval_euclidean_fit_regression.py --dataset all \
  --baseline-report rust_umap/benchmarks/current_fit_regression.json
```

`eval_inverse_quality.py` now fails on asymmetric success/failure by default to avoid biased comparisons.
