# rust_umap

A ground-up Rust implementation of the core UMAP pipeline for reproducible algorithm study.

## What is implemented

- KNN search (Euclidean):
  - Exact all-pairs baseline
  - Optional NN-Descent-style approximate mode for larger datasets
- Smooth KNN distance estimation (`sigma`, `rho`) with binary search
- Fuzzy simplicial set construction and symmetric fuzzy union
- Automatic `(a, b)` curve parameter fitting from `(spread, min_dist)`
- Two initialization strategies:
  - `InitMethod::Spectral` (normalized graph Laplacian eigenvectors)
  - `InitMethod::Random`
- Stochastic layout optimization with negative sampling
- End-to-end training APIs:
  - `fit_transform`
  - `fit`
- New point embedding API:
  - `transform`
- Approximate inverse mapping API:
  - `inverse_transform` (Euclidean metric)

## What is intentionally out of scope (for this version)

- Full `pynndescent`-equivalent ANN quality/performance parity
- Non-Euclidean metrics
- Sparse input support
- Multi-component spectral layout heuristics from upstream implementation

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
2. Check for euclidean no-regression on timing and memory.
3. Run a deeper optimization-stage benchmark report when explicitly requested.

The Rust crate remains the unit under test in all three stages.

## Minimal library usage

```rust
use rust_umap::{InitMethod, UmapModel, UmapParams};

let data: Vec<Vec<f32>> = vec![
    vec![0.0, 0.1, 0.2],
    vec![0.2, 0.1, 0.0],
    // ...
];

let mut model = UmapModel::new(UmapParams {
    init: InitMethod::Spectral,
    ..UmapParams::default()
});

let embedding = model.fit_transform(&data)?;
let new_points = vec![vec![0.1, 0.1, 0.2]];
let transformed = model.transform(&new_points)?;
let reconstructed = model.inverse_transform(&transformed)?;
```

## Notes on fidelity

The core equations for graph construction and layout optimization are aligned with standard UMAP behavior.
This implementation prioritizes clarity and reproducibility over large-scale performance.
