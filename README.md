# umap-rs

A Rust-first UMAP implementation with reproducible cross-implementation benchmarking against Python `umap-learn` and R `uwot`.

## Repository layout

- `rust_umap/`: Rust UMAP library and CLI binaries.
- `benchmarks/`: Fair benchmark harness and reports.
- `UMAP_MATHEMATICAL_DOCUMENTATION*.md`: mathematical notes.

## Quick start

```bash
cd rust_umap
cargo build --release
cargo test
```

## Fair benchmark report

The latest fairness-controlled real-dataset comparison report is available at:

- `benchmarks/report_real_fair.md`
- `benchmarks/report_real_fair.json`

## Benchmark dependencies

The benchmark scripts use:

- Python: `umap-learn`, `numpy`, `scikit-learn`, `scipy`
- R: `uwot`, `jsonlite`

See `benchmarks/compare_real_impls_fair.py` for exact execution settings.
