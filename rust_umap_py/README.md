# rust-umap-py

Python bindings for `rust_umap` built with PyO3 + maturin.

## Local build

```bash
. .venv/bin/activate 2>/dev/null || {
  python -m venv .venv
  . .venv/bin/activate
}
python -m pip install --upgrade pip maturin
maturin develop --manifest-path rust_umap_py/Cargo.toml
```

## Quick usage

```python
import numpy as np
from rust_umap_py import Umap

x = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
model = Umap(n_neighbors=15, n_components=2, n_epochs=120, random_seed=42, init="random")
emb = model.fit_transform(x)
```
