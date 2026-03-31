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

## Ecosystem Integration (Python Binding MVP)

This repository now includes a Python binding entrypoint under `rust_umap_py/`
for cross-ecosystem usage and fair benchmarking against `umap-learn`.

### Build and install binding locally

```bash
python -m pip install --upgrade pip maturin
maturin develop --manifest-path rust_umap_py/Cargo.toml
```

### End-to-end call example (library API)

```bash
python - <<'PY'
import numpy as np
from rust_umap_py import Umap

rng = np.random.default_rng(42)
x = rng.normal(size=(400, 16)).astype(np.float32)

model = Umap(
    n_neighbors=15,
    n_components=2,
    n_epochs=120,
    metric="euclidean",
    random_seed=42,
    init="random",
)
emb = model.fit_transform(x)
print("embedding shape:", emb.shape, "dtype:", emb.dtype)
PY
```

### Ecosystem benchmark (umap-learn vs rust_umap_py)

```bash
python benchmarks/compare_ecosystem_python_binding.py \
  --python-bin python \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000
```

Outputs:
- `benchmarks/report_ecosystem_python_binding.json`
- `benchmarks/report_ecosystem_python_binding.md`

## CI and Benchmark Gates

The repository CI is intentionally staged:

1. Public-implementation consistency smoke check.
2. No-regression smoke check against a baseline branch (metric matrix: euclidean/manhattan/cosine).
3. Ecosystem Python binding smoke + machine-readable gate.
4. Optional optimization-stage benchmark report in a deeper manual/scheduled workflow.

Fast PR validation lives in `.github/workflows/ci.yml`.
Deeper benchmark reporting lives in `.github/workflows/deep-benchmark-report.yml`.

## Local Validation Commands

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "python3/python not found" >&2
  exit 1
fi

cargo test --manifest-path rust_umap/Cargo.toml

$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r benchmarks/requirements-bench.txt pytest maturin

$PYTHON_BIN -m py_compile \
  benchmarks/compare_real_impls_fair.py \
  benchmarks/compare_ecosystem_python_binding.py \
  benchmarks/ci_consistency_smoke.py \
  benchmarks/ci_ann_smoke.py \
  benchmarks/ci_no_regression.py \
  benchmarks/run_rust_umap_py.py \
  benchmarks/run_rust_umap_py_algo.py

$PYTHON_BIN benchmarks/ci_ann_smoke.py \
  --python-bin "$PYTHON_BIN" \
  --rscript-bin ""

if command -v Rscript >/dev/null 2>&1; then
  Rscript benchmarks/install_r_bench_deps.R
  $PYTHON_BIN benchmarks/ci_consistency_smoke.py \
    --python-bin "$PYTHON_BIN" \
    --rscript-bin Rscript \
    --require-r
else
  $PYTHON_BIN benchmarks/ci_consistency_smoke.py \
    --python-bin "$PYTHON_BIN" \
    --rscript-bin ""
fi

# Build/install local binding before running binding tests.
maturin develop --manifest-path rust_umap_py/Cargo.toml

# candidate-root and baseline-root must point to different trees.
git worktree add ../umap-rs-baseline HEAD~1
$PYTHON_BIN benchmarks/ci_no_regression.py \
  --candidate-root "$PWD" \
  --baseline-root "$(cd ../umap-rs-baseline && pwd)"
git worktree remove ../umap-rs-baseline

$PYTHON_BIN -m pytest -q rust_umap_py/tests/test_binding.py
```
