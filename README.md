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

## CI and Benchmark Gates

The repository CI is intentionally staged:

1. Public-implementation consistency smoke check.
2. Euclidean no-regression smoke check against a baseline branch.
3. Optional optimization-stage benchmark report in a deeper manual/scheduled workflow.

Fast PR validation lives in `.github/workflows/ci.yml`.
Deeper benchmark reporting lives in `.github/workflows/deep-benchmark-report.yml`.

## Local Validation Commands

```bash
cargo test --manifest-path rust_umap/Cargo.toml

python3 -m py_compile \
  benchmarks/compare_real_impls_fair.py \
  benchmarks/ci_consistency_smoke.py \
  benchmarks/ci_no_regression.py

python benchmarks/ci_consistency_smoke.py \
  --python-bin python \
  --rscript-bin Rscript \
  --require-r

python benchmarks/ci_no_regression.py \
  --candidate-root . \
  --baseline-root .
```
