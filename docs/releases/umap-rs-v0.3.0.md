# umap-rs 0.3.0

`umap-rs 0.3.0` is the first PyPI release published under the unified package
name `umap-rs`, with the Python import path `umap_rs`.

## Highlights

- Unified Python naming:
  - `pip install umap-rs`
  - `import umap_rs`
- Shipped typed package assets for the public API, including stubs,
  `py.typed`, `UmapKwargs`, and improved docstrings/signature help.
- Kept the API layered:
  - `Umap` and `fit_transform(...)` are the default path
  - `fit_transform_with_knn(...)` remains the advanced precomputed-kNN path
- Published wheels for Linux x86_64, Windows x86_64, and macOS arm64 on
  Python 3.9-3.13, plus an sdist.

## Install

```bash
pip install umap-rs
```

```python
from umap_rs import Umap, fit_transform
```

## Migration

If you were using an earlier local build or draft package setup, update your
installation and imports to the unified names:

```bash
pip install umap-rs
```

```python
import umap_rs
```

## Python Support

- Supported: Python 3.9 through 3.13
- Python 3.14 was probed from sdist in CI, but is not yet a declared supported
  target for this release

## Notes

- The binding remains intentionally thin: Python handles input normalization,
  while Rust owns validation and compute-heavy paths whenever practical.
- The documented quickstart path is still `Umap` or `fit_transform(...)`.
