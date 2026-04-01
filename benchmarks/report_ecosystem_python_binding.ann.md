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
- california_housing: n_used=15000, n_original=20640, d=8

## Group A: e2e_mixed_knn_strategy

- This group intentionally keeps each implementation's runtime defaults; strategy may differ across implementations/datasets.
- python_umap_learn: exact kNN (`force_approximation_algorithm=False`); rust_umap_py: adaptive (`use_approximate_knn=True`, threshold=4096).

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.208 ± 0.000 | 345.3 ± 0.0 |
| rust_umap_py | 0.182 ± 0.000 | 30.2 ± 0.0 |

- knn_strategy_equivalence: strict_exact
- knn_strategy_note: Both implementations used exact kNN for this dataset.
- python_umap_learn strategy: exact
- rust_umap_py strategy: exact
- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916196
  - rust_umap_py: 0.914304
- original_knn_recall@15:
  - python_umap_learn: 0.398711
  - rust_umap_py: 0.406327
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.041796, knn_overlap@15=0.608787

### Dataset: digits

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 6.562 ± 0.000 | 358.8 ± 0.0 |
| rust_umap_py | 0.491 ± 0.000 | 33.1 ± 0.0 |

- knn_strategy_equivalence: strict_exact
- knn_strategy_note: Both implementations used exact kNN for this dataset.
- python_umap_learn strategy: exact
- rust_umap_py strategy: exact
- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.962690
  - rust_umap_py: 0.967399
- original_knn_recall@15:
  - python_umap_learn: 0.456000
  - rust_umap_py: 0.465944
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.908825, knn_overlap@15=0.625111

### Dataset: california_housing

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 12.791 ± 0.000 | 470.2 ± 0.0 |
| rust_umap_py | 3.400 ± 0.000 | 60.1 ± 0.0 |

- knn_strategy_equivalence: not_equivalent
- knn_strategy_note: Python path stays exact (force_approximation_algorithm=False), while rust_umap_py switched to ANN because n_samples exceeded approx_knn_threshold.
- python_umap_learn strategy: exact
- rust_umap_py strategy: approximate_ann
- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.957255
  - rust_umap_py: 0.961256
- original_knn_recall@15:
  - python_umap_learn: 0.405444
  - rust_umap_py: 0.418222
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.665226, knn_overlap@15=0.526722

## Group B: algo_exact_shared_knn_exact

- This group enforces strict comparability: both implementations use the same precomputed exact shared kNN graph.

### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.315 ± 0.000 | 309.2 |
| rust_umap_py | 0.120 ± 0.000 | 30.8 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916197
  - rust_umap_py: 0.922518
- original_knn_recall@15:
  - python_umap_learn: 0.395079
  - rust_umap_py: 0.412302
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.206557, knn_overlap@15=0.495958

### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.925 ± 0.000 | 311.9 |
| rust_umap_py | 0.338 ± 0.000 | 34.1 |

- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.962798
  - rust_umap_py: 0.969112
- original_knn_recall@15:
  - python_umap_learn: 0.462778
  - rust_umap_py: 0.468889
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.858440, knn_overlap@15=0.612389

### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 5.254 ± 0.000 | 328.9 |
| rust_umap_py | 2.989 ± 0.000 | 62.6 |

- sample_size_for_consistency: 1200
- trustworthiness@15:
  - python_umap_learn: 0.961392
  - rust_umap_py: 0.952556
- original_knn_recall@15:
  - python_umap_learn: 0.422833
  - rust_umap_py: 0.413778
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.487096, knn_overlap@15=0.677111

## Interop Audit

- Input dtype is normalized to float32 before crossing Python/Rust boundary.
- kNN indices use int64 and kNN distances use float32 in the binding path.
- Thread counts are pinned to 1 for BLAS/OpenMP/Numba to avoid cross-runtime thread bias.
- Random seed is aligned across implementations with seed=42 by default.
- Algorithm timing scope in both bindings is aligned to the fit_transform* call only.
- No post-fit dtype conversion/copy is included in algorithm timers on either side.
- Current rust_umap core stores row-major Vec<Vec<f32>>, so one boundary copy is still required.
