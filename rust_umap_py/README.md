# rust-umap-py

Python bindings for `rust_umap` built with PyO3 + maturin.

Version `0.3.0` focuses on IDE-help quality for the public Python API:

- the exported surface has useful type hints
- public methods carry docstrings that explain inputs and outputs
- `help(rust_umap_py.Umap)` and editor hover should be informative

The binding remains intentionally thin: Python normalizes arrays and CSR inputs,
while Rust owns validation and compute-heavy paths whenever practical.

## Local build

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
uv venv --python "$PYTHON_BIN" .venv
uv pip install --python .venv/bin/python --upgrade pip maturin
uv run --python .venv/bin/python maturin develop --manifest-path rust_umap_py/Cargo.toml
uv run --python .venv/bin/python python -I -m pytest -q rust_umap_py/tests/test_binding.py
```

## Quick usage

```python
import numpy as np
from rust_umap_py import Umap

x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
model = Umap(n_neighbors=15, n_components=2, n_epochs=120, random_seed=42, init="random")
emb = model.fit_transform(x)
```

## API layers

### Main API

Most users should start here:

- `Umap`
- `fit_transform`
- `Umap.fit`
- `Umap.fit_transform`
- `Umap.transform`
- `Umap.inverse_transform`

These methods accept NumPy arrays by default and support the documented `out=`
buffers where available.

### Advanced API

`Umap.fit_transform_with_knn(...)` is available for callers who already have a
precomputed exact or shared kNN graph. It is useful for benchmarks and
parameter sweeps, but it is not the recommended first-stop quickstart.
