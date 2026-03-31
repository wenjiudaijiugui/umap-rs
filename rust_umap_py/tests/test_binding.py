import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

# Ensure we import the installed package, not the repo's top-level
# `rust_umap_py/` crate directory as a namespace package.
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT]

from rust_umap_py import Umap


def make_dataset(n_samples: int = 180, n_features: int = 12, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.stack(
        [
            np.linspace(-2.0, 2.0, n_features, dtype=np.float32),
            np.linspace(1.5, -1.5, n_features, dtype=np.float32),
            np.zeros(n_features, dtype=np.float32),
        ]
    )
    labels = rng.integers(0, len(centers), size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.35, size=(n_samples, n_features)).astype(np.float32)
    x = centers[labels] + noise
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def test_fit_transform_out_buffer_and_inverse_roundtrip() -> None:
    x = make_dataset()
    model = Umap(
        n_neighbors=12,
        n_components=2,
        n_epochs=80,
        metric="euclidean",
        init="random",
        random_seed=7,
        use_approximate_knn=False,
    )

    out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform(x, out=out)

    assert emb is out
    assert emb.dtype == np.float32
    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))

    query = x[:24]
    transformed = model.transform(query)
    reconstructed = model.inverse_transform(transformed)

    assert transformed.shape == (query.shape[0], 2)
    assert reconstructed.shape == query.shape
    assert np.all(np.isfinite(transformed))
    assert np.all(np.isfinite(reconstructed))


def test_precomputed_knn_path_consistency() -> None:
    x = make_dataset(n_samples=160, n_features=10, seed=123)

    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=1)
    nbrs.fit(x)
    dists, idx = nbrs.kneighbors(x)
    knn_idx = idx[:, 1 : k + 1].astype(np.int64)
    knn_dist = dists[:, 1 : k + 1].astype(np.float32)

    base_params = dict(
        n_neighbors=k,
        n_components=2,
        n_epochs=60,
        metric="euclidean",
        init="random",
        random_seed=11,
        use_approximate_knn=False,
    )
    model_direct = Umap(**base_params)
    model_knn = Umap(**base_params)

    emb_direct = model_direct.fit_transform(x)
    emb_knn = model_knn.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")

    assert emb_direct.shape == emb_knn.shape
    assert np.all(np.isfinite(emb_direct))
    assert np.all(np.isfinite(emb_knn))

    trust_direct = float(trustworthiness(x, emb_direct, n_neighbors=k))
    trust_knn = float(trustworthiness(x, emb_knn, n_neighbors=k))
    assert abs(trust_direct - trust_knn) < 0.05


def test_non_float32_and_non_contiguous_input_is_normalized() -> None:
    x = make_dataset(n_samples=100, n_features=8, seed=5).astype(np.float64)
    x_fortran = np.asfortranarray(x)

    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        metric="euclidean",
        init="random",
        random_seed=13,
        use_approximate_knn=False,
    )

    emb = model.fit_transform(x_fortran)
    assert emb.dtype == np.float32
    assert emb.flags.c_contiguous
    assert emb.shape == (x_fortran.shape[0], 2)


def test_zero_column_data_is_rejected_with_value_error() -> None:
    x = np.empty((16, 0), dtype=np.float32)
    model = Umap(
        n_neighbors=5,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=3,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="data must have at least one column"):
        model.fit(x)

    with pytest.raises(ValueError, match="data must have at least one column"):
        model.fit_transform(x)


def test_zero_column_precomputed_knn_is_rejected_with_value_error() -> None:
    x = make_dataset(n_samples=40, n_features=6, seed=9)
    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=19,
        use_approximate_knn=False,
    )

    knn_idx = np.empty((x.shape[0], 0), dtype=np.int64)
    knn_dist = np.empty((x.shape[0], 0), dtype=np.float32)

    with pytest.raises(ValueError, match="knn columns must be >= n_neighbors"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")
