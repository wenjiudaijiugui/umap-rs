from __future__ import annotations

from typing import Any

import numpy as np

from ._rust_umap_py import UmapCore


def _as_f32_matrix(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    return arr


def _as_knn_indices(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64, order="C")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    return arr


def _maybe_as_csr_parts(x: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    if getattr(x, "format", None) != "csr":
        return None

    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a 2D CSR matrix")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_cols <= 0:
        raise ValueError(f"{name} must have at least one column")

    indptr = np.asarray(x.indptr, dtype=np.int64, order="C")
    indices = np.asarray(x.indices, dtype=np.int64, order="C")
    data = np.asarray(x.data, dtype=np.float32, order="C")
    if indptr.ndim != 1 or indices.ndim != 1 or data.ndim != 1:
        raise ValueError(f"{name} CSR arrays must be 1D")
    if indices.shape[0] != data.shape[0]:
        raise ValueError(f"{name} CSR indices/data length mismatch")
    return indptr, indices, data, n_rows, n_cols


def _validate_precomputed_shapes(
    data: np.ndarray,
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    n_neighbors: int,
) -> None:
    if knn_indices.shape != knn_dists.shape:
        raise ValueError("knn_indices and knn_dists must have identical shapes")
    if knn_indices.shape[0] != data.shape[0]:
        raise ValueError("knn row count must match data row count")
    if knn_indices.shape[1] < n_neighbors:
        raise ValueError(f"knn columns must be >= n_neighbors ({n_neighbors})")


def _validate_precomputed_values(knn_indices: np.ndarray, knn_dists: np.ndarray) -> None:
    if np.any(knn_indices < 0):
        raise ValueError("knn indices must be non-negative integers")
    if not np.all(np.isfinite(knn_dists)):
        raise ValueError("knn_dists must contain only finite values")
    if np.any(knn_dists < 0):
        raise ValueError("knn_dists must be non-negative")


def _as_out_buffer(out: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy ndarray")
    if out.dtype != np.float32:
        raise TypeError(f"out dtype must be float32, got {out.dtype}")
    if out.ndim != 2:
        raise ValueError(f"out must be 2D, got ndim={out.ndim}")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if not out.flags.writeable:
        raise ValueError("out must be writeable")
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"output buffer shape mismatch: expected {shape}, got {tuple(out.shape)}")
    return out


class Umap:
    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        n_components: int = 2,
        n_epochs: int | None = None,
        metric: str = "euclidean",
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        local_connectivity: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        random_seed: int = 42,
        init: str = "spectral",
        use_approximate_knn: bool = True,
        approx_knn_candidates: int = 30,
        approx_knn_iters: int = 10,
        approx_knn_threshold: int = 4096,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self._n_features: int | None = None
        self._core = UmapCore(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            metric=metric,
            learning_rate=learning_rate,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            set_op_mix_ratio=set_op_mix_ratio,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            random_seed=random_seed,
            init=init,
            use_approximate_knn=use_approximate_knn,
            approx_knn_candidates=approx_knn_candidates,
            approx_knn_iters=approx_knn_iters,
            approx_knn_threshold=approx_knn_threshold,
        )

    def fit(self, data: Any) -> "Umap":
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            indptr, indices, values, _, n_cols = csr
            self._core.fit_sparse_csr(indptr, indices, values, n_cols)
            self._n_features = n_cols
            return self

        arr = _as_f32_matrix(data, "data")
        self._core.fit(arr)
        self._n_features = arr.shape[1]
        return self

    def fit_transform(self, data: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        csr = _maybe_as_csr_parts(data, "data")
        if csr is not None:
            indptr, indices, values, n_rows, n_cols = csr
            self._n_features = n_cols
            expected_shape = (n_rows, self.n_components)
            if out is None:
                return self._core.fit_transform_sparse_csr(indptr, indices, values, n_cols)
            out_buf = _as_out_buffer(out, expected_shape)
            self._core.fit_transform_sparse_csr_into(indptr, indices, values, n_cols, out_buf)
            return out_buf

        arr = _as_f32_matrix(data, "data")
        self._n_features = arr.shape[1]
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.fit_transform(arr)
        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_into(arr, out_buf)
        return out_buf

    def fit_transform_with_knn(
        self,
        data: Any,
        knn_indices: Any,
        knn_dists: Any,
        *,
        knn_metric: str = "euclidean",
        validate_precomputed: bool = True,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = _as_f32_matrix(data, "data")
        idx = _as_knn_indices(knn_indices, "knn_indices")
        dist = _as_f32_matrix(knn_dists, "knn_dists")
        _validate_precomputed_shapes(arr, idx, dist, self.n_neighbors)
        if validate_precomputed:
            _validate_precomputed_values(idx, dist)

        self._n_features = arr.shape[1]
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.fit_transform_with_knn(arr, idx, dist, knn_metric)

        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_with_knn_into(arr, idx, dist, out_buf, knn_metric)
        return out_buf

    def transform(self, query: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        arr = _as_f32_matrix(query, "query")
        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.transform(arr)
        out_buf = _as_out_buffer(out, expected_shape)
        self._core.transform_into(arr, out_buf)
        return out_buf

    def inverse_transform(self, embedded_query: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        arr = _as_f32_matrix(embedded_query, "embedded_query")
        if out is None:
            return self._core.inverse_transform(arr)
        if self._n_features is None:
            raise RuntimeError("model must be fit before inverse_transform(out=...)")
        out_buf = _as_out_buffer(out, (arr.shape[0], self._n_features))
        self._core.inverse_transform_into(arr, out_buf)
        return out_buf


def fit_transform(data: Any, **kwargs: Any) -> np.ndarray:
    model = Umap(**kwargs)
    return model.fit_transform(data)
