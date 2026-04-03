# Changelog

All notable changes to this repository are documented in this file.

The format follows Keep a Changelog and uses semantic-versioned release headings
for user-facing milestones.

## [0.3.0] - 2026-04-03

### Highlights

- Shipped typed Python package assets for `umap_rs`, including
  `__init__.pyi`, `_api.pyi`, and `py.typed`, plus `UmapKwargs`, richer
  docstrings, hover text, and manual smoke examples for the public API.
- Documented the Python API in layers: dense `Umap` and one-shot
  `fit_transform` are the default path, while
  `Umap.fit_transform_with_knn(...)` remains an advanced precomputed-kNN
  interface.
- Standardized public Python keyword naming on `**kwargs` and removed the
  overlapping stub pattern that caused Pyright/Pylance unreachable-overload
  diagnostics.
- Updated repository and package docs for the 0.3.0 Python surface and current
  scope boundary.
- Added release metadata, a Python release workflow, and verified the binding
  with `maturin build --release --sdist`, `twine check`, and clean
  wheel-install smoke tests.
