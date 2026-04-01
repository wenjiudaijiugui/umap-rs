# Ecosystem Python Binding Benchmark Report

## Setup
- n_neighbors=15, n_components=2, n_epochs=200, init=random, metric=euclidean, seed=42
- warmup=0, repeats=1, python_bin=python
- implementations: python_umap_learn, rust_umap_py
- groups: e2e_mixed_knn_strategy, algo_exact_shared_knn_exact
- thread pinning: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONHASHSEED=0

## Datasets

- breast_cancer: n_used=569, n_original=569, d=30
- digits: n_used=1797, n_original=1797, d=64
- california_housing: n_used=4097, n_original=20640, d=8

## Group A: e2e_mixed_knn_strategy

- This group intentionally keeps each implementation's runtime defaults; strategy may differ across implementations/datasets.
- python_umap_learn: exact kNN (`force_approximation_algorithm=False`); rust_umap_py: adaptive (`use_approximate_knn=True`, threshold=4096).

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 7.175 ± 0.000 | 269.1 ± 0.0 |
| rust_umap_py | 0.951 ± 0.000 | 37.9 ± 0.0 |

- knn_strategy_equivalence: strict_exact
- knn_strategy_note: Both implementations used exact kNN for this dataset.
- python_umap_learn strategy: exact
- rust_umap_py strategy: exact
- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.914866
  - rust_umap_py: 0.914375
- original_knn_recall@15:
  - python_umap_learn: 0.408084
  - rust_umap_py: 0.405155
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.070478, knn_overlap@15=0.573052

### Dataset: digits

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.733 ± 0.000 | 280.0 ± 0.0 |
| rust_umap_py | 5.229 ± 0.000 | 40.8 ± 0.0 |

- knn_strategy_equivalence: strict_exact
- knn_strategy_note: Both implementations used exact kNN for this dataset.
- python_umap_learn strategy: exact
- rust_umap_py strategy: exact
- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.967994
  - rust_umap_py: 0.966510
- original_knn_recall@15:
  - python_umap_learn: 0.462111
  - rust_umap_py: 0.468944
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.692903, knn_overlap@15=0.691222

### Dataset: california_housing

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 10.961 ± 0.000 | 305.9 ± 0.0 |
| rust_umap_py | 7.432 ± 0.000 | 48.3 ± 0.0 |

- knn_strategy_equivalence: not_equivalent
- knn_strategy_note: Python path stays exact (force_approximation_algorithm=False), while rust_umap_py switched to ANN because n_samples exceeded approx_knn_threshold.
- python_umap_learn strategy: exact
- rust_umap_py strategy: approximate_ann
- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.961251
  - rust_umap_py: 0.953599
- original_knn_recall@15:
  - python_umap_learn: 0.415389
  - rust_umap_py: 0.406500
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.296875, knn_overlap@15=0.572833

## Group B: algo_exact_shared_knn_exact

- This group enforces strict comparability: both implementations use the same precomputed exact shared kNN graph.

### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.457 ± 0.000 | 246.0 |
| rust_umap_py | 0.719 ± 0.000 | 38.5 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.917910
  - rust_umap_py: 0.916365
- original_knn_recall@15:
  - python_umap_learn: 0.397891
  - rust_umap_py: 0.410076
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.171682, knn_overlap@15=0.543995

### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.821 ± 0.000 | 248.7 |
| rust_umap_py | 2.057 ± 0.000 | 41.8 |

- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.967898
  - rust_umap_py: 0.965782
- original_knn_recall@15:
  - python_umap_learn: 0.467722
  - rust_umap_py: 0.463111
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.929708, knn_overlap@15=0.666833

### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 2.416 ± 0.000 | 251.4 |
| rust_umap_py | 4.969 ± 0.000 | 49.3 |

- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.964544
  - rust_umap_py: 0.958689
- original_knn_recall@15:
  - python_umap_learn: 0.416667
  - rust_umap_py: 0.413111
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.220417, knn_overlap@15=0.619944

## Interop Audit

- Input dtype is normalized to float32 before crossing Python/Rust boundary.
- kNN indices use int64 and kNN distances use float32 in the binding path.
- Thread counts are pinned to 1 for BLAS/OpenMP/Numba to avoid cross-runtime thread bias.
- Random seed is aligned across implementations with seed=42 by default.
- Algorithm timing scope in both bindings is aligned to the fit_transform* call only.
- No post-fit dtype conversion/copy is included in algorithm timers on either side.
- Current rust_umap core stores row-major Vec<Vec<f32>>, so one boundary copy is still required.
