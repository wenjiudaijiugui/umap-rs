import numpy as np

# Ensure we import the installed package, not the repo's top-level
# `rust_umap_py/` crate directory as a namespace package.
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _REPO_ROOT]

from rust_umap_py import Umap
import rust_umap_py._api as api


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


def test_ann_mode_overrides_legacy_knn_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeCore:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.n_features = None

    monkeypatch.setattr(api, "UmapCore", FakeCore)

    auto = api.Umap(
        ann_mode="auto",
        use_approximate_knn=False,
        approx_knn_threshold=321,
    )
    exact = api.Umap(
        ann_mode="exact",
        use_approximate_knn=True,
        approx_knn_threshold=321,
    )
    approximate = api.Umap(
        ann_mode="approximate",
        use_approximate_knn=False,
        approx_knn_threshold=321,
    )

    assert auto.ann_mode == "auto"
    assert exact.ann_mode == "exact"
    assert approximate.ann_mode == "approximate"
    assert captured[0]["use_approximate_knn"] is False
    assert captured[0]["approx_knn_threshold"] == 321
    assert captured[1]["use_approximate_knn"] is False
    assert captured[1]["approx_knn_threshold"] == 321
    assert captured[2]["use_approximate_knn"] is True
    assert captured[2]["approx_knn_threshold"] == 0


def test_ann_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported ann_mode 'hybrid'"):
        Umap(ann_mode="hybrid")


def test_transform_and_inverse_transform_support_out_buffers() -> None:
    x = make_dataset(n_samples=120, n_features=10, seed=21)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=60,
        metric="euclidean",
        init="random",
        random_seed=17,
        use_approximate_knn=False,
    )
    model.fit(x)

    query = x[:18]
    transformed_out = np.empty((query.shape[0], 2), dtype=np.float32)
    transformed = model.transform(query, out=transformed_out)
    assert transformed is transformed_out
    assert np.all(np.isfinite(transformed))

    reconstructed_out = np.empty((query.shape[0], x.shape[1]), dtype=np.float32)
    reconstructed = model.inverse_transform(transformed, out=reconstructed_out)
    assert reconstructed is reconstructed_out
    assert reconstructed.shape == query.shape
    assert np.all(np.isfinite(reconstructed))


def test_inverse_transform_empty_input_preserves_feature_width() -> None:
    x = make_dataset(n_samples=96, n_features=11, seed=25)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=50,
        metric="euclidean",
        init="random",
        random_seed=31,
        use_approximate_knn=False,
    )
    model.fit(x)

    empty_embedded = np.empty((0, 2), dtype=np.float32)
    reconstructed = model.inverse_transform(empty_embedded)
    assert reconstructed.shape == (0, x.shape[1])

    out = np.empty((0, x.shape[1]), dtype=np.float32)
    reconstructed_out = model.inverse_transform(empty_embedded, out=out)
    assert reconstructed_out is out
    assert reconstructed_out.shape == (0, x.shape[1])


def test_precomputed_knn_path_consistency() -> None:
    trustworthiness = pytest.importorskip("sklearn.manifold").trustworthiness
    nearest_neighbors = pytest.importorskip("sklearn.neighbors")
    NearestNeighbors = nearest_neighbors.NearestNeighbors

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


def test_precomputed_knn_rejects_non_finite_distances_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=31)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)
    knn_dist[0, 0] = np.nan

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=5,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn_dists must contain only finite values"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_unsupported_metric_and_init_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported metric 'chebyshev'"):
        Umap(metric="chebyshev")

    with pytest.raises(ValueError, match="unsupported init 'pca'"):
        Umap(init="pca")


def test_precomputed_knn_rejects_shape_mismatch_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=37)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_dist = np.ones((x.shape[0], k - 1), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=7,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn_indices and knn_dists must have identical shapes"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


def test_precomputed_knn_rejects_negative_indices_early() -> None:
    x = make_dataset(n_samples=48, n_features=6, seed=41)
    k = 8
    knn_idx = np.tile(np.arange(k, dtype=np.int64), (x.shape[0], 1))
    knn_idx[0, 0] = -1
    knn_dist = np.ones((x.shape[0], k), dtype=np.float32)

    model = Umap(
        n_neighbors=k,
        n_components=2,
        n_epochs=20,
        init="random",
        random_seed=9,
        use_approximate_knn=False,
    )

    with pytest.raises(ValueError, match="knn indices must be non-negative integers"):
        model.fit_transform_with_knn(x, knn_idx, knn_dist, knn_metric="euclidean")


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


def test_fit_transform_rejects_invalid_out_buffers() -> None:
    x = make_dataset(n_samples=96, n_features=8, seed=15)
    model = Umap(
        n_neighbors=10,
        n_components=2,
        n_epochs=40,
        metric="euclidean",
        init="random",
        random_seed=27,
        use_approximate_knn=False,
    )

    with pytest.raises(TypeError, match="out dtype must be float32"):
        bad_dtype = np.empty((x.shape[0], 2), dtype=np.float64)
        model.fit_transform(x, out=bad_dtype)

    with pytest.raises(ValueError, match="out must be C-contiguous"):
        bad_order = np.empty((x.shape[0], 2), dtype=np.float32, order="F")
        model.fit_transform(x, out=bad_order)

    with pytest.raises(ValueError, match="out must be writeable"):
        readonly = np.empty((x.shape[0], 2), dtype=np.float32)
        readonly.setflags(write=False)
        model.fit_transform(x, out=readonly)


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


def test_sparse_csr_fit_transform_and_dense_transform_out_buffer() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=90, n_features=14, seed=55)
    x[x < 0.15] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=8,
        n_components=2,
        n_epochs=50,
        metric="cosine",
        init="random",
        random_seed=23,
        use_approximate_knn=False,
    )

    emb_out = np.empty((x.shape[0], 2), dtype=np.float32)
    emb = model.fit_transform(x_csr, out=emb_out)
    assert emb is emb_out
    assert emb.shape == (x.shape[0], 2)
    assert np.all(np.isfinite(emb))

    query = x[:12]
    transformed_out = np.empty((query.shape[0], 2), dtype=np.float32)
    transformed = model.transform(query, out=transformed_out)
    assert transformed is transformed_out
    assert np.all(np.isfinite(transformed))


def test_sparse_csr_fit_tracks_feature_count_for_inverse_out_error() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    x = make_dataset(n_samples=64, n_features=9, seed=71)
    x[x < 0.2] = 0.0
    x_csr = scipy_sparse.csr_matrix(x)

    model = Umap(
        n_neighbors=6,
        n_components=2,
        n_epochs=40,
        metric="manhattan",
        init="random",
        random_seed=29,
        use_approximate_knn=False,
    )
    model.fit(x_csr)

    with pytest.raises(ValueError, match="output buffer shape mismatch"):
        bad_out = np.empty((5, x.shape[1] + 1), dtype=np.float32)
        model.inverse_transform(np.zeros((5, 2), dtype=np.float32), out=bad_out)
