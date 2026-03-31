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
        raise ValueError(f"out shape mismatch: expected {shape}, got {tuple(out.shape)}")
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
        self.n_components = int(n_components)
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
        arr = _as_f32_matrix(data, "data")
        self._core.fit(arr)
        return self

    def fit_transform(self, data: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        arr = _as_f32_matrix(data, "data")
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
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = _as_f32_matrix(data, "data")
        idx = _as_knn_indices(knn_indices, "knn_indices")
        dist = _as_f32_matrix(knn_dists, "knn_dists")
        if idx.shape != dist.shape:
            raise ValueError("knn_indices and knn_dists must have identical shapes")
        if idx.shape[0] != arr.shape[0]:
            raise ValueError("knn row count must match data row count")

        expected_shape = (arr.shape[0], self.n_components)
        if out is None:
            return self._core.fit_transform_with_knn(arr, idx, dist, knn_metric)

        out_buf = _as_out_buffer(out, expected_shape)
        self._core.fit_transform_with_knn_into(arr, idx, dist, out_buf, knn_metric)
        return out_buf

    def transform(self, query: Any) -> np.ndarray:
        arr = _as_f32_matrix(query, "query")
        return self._core.transform(arr)

    def inverse_transform(self, embedded_query: Any) -> np.ndarray:
        arr = _as_f32_matrix(embedded_query, "embedded_query")
        return self._core.inverse_transform(arr)


def fit_transform(data: Any, **kwargs: Any) -> np.ndarray:
    model = Umap(**kwargs)
    return model.fit_transform(data)
