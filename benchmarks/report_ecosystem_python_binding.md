# Ecosystem Python Binding Benchmark Report

## Setup
- n_neighbors=15, n_components=2, n_epochs=200, init=random, metric=euclidean, seed=42
- warmup=1, repeats=2, python_bin=/home/shenshang/miniforge3/envs/umap_bench/bin/python
- implementations: python_umap_learn, rust_umap_py
- thread pinning: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONHASHSEED=0

## Datasets

- breast_cancer: n_used=569, n_original=569, d=30
- digits: n_used=1797, n_original=1797, d=64

## Group A: e2e_default_ann

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.438 ± 0.360 | 346.1 ± 0.8 |
| rust_umap_py | 0.968 ± 0.001 | 31.5 ± 0.0 |

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
| python_umap_learn | 6.668 ± 0.143 | 358.9 ± 0.2 |
| rust_umap_py | 5.298 ± 0.077 | 34.4 ± 0.0 |

- sample_size_for_consistency: 1500
- trustworthiness@15:
  - python_umap_learn: 0.963175
  - rust_umap_py: 0.969673
- original_knn_recall@15:
  - python_umap_learn: 0.440533
  - rust_umap_py: 0.457067
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.925750, knn_overlap@15=0.583867

## Group B: algo_exact_shared_knn

### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.203 ± 0.003 | 309.0 |
| rust_umap_py | 1.041 ± 0.039 | 32.8 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916197
  - rust_umap_py: 0.920500
- original_knn_recall@15:
  - python_umap_learn: 0.395079
  - rust_umap_py: 0.411482
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.218503, knn_overlap@15=0.485179

### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.832 ± 0.009 | 312.1 |
| rust_umap_py | 2.615 ± 0.022 | 35.4 |

- sample_size_for_consistency: 1500
- trustworthiness@15:
  - python_umap_learn: 0.962286
  - rust_umap_py: 0.967911
- original_knn_recall@15:
  - python_umap_learn: 0.440222
  - rust_umap_py: 0.448311
- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity=0.892945, knn_overlap@15=0.580178

## Interop Audit

- Input dtype is normalized to float32 before crossing Python/Rust boundary.
- kNN indices use int64 and kNN distances use float32 in the binding path.
- Thread counts are pinned to 1 for BLAS/OpenMP/Numba to avoid cross-runtime thread bias.
- Random seed is aligned across implementations with seed=42 by default.
- rust_umap_py algorithm benchmark uses reusable output buffer to reduce allocation overhead and peak RSS.
- Current rust_umap core stores row-major Vec<Vec<f32>>, so one boundary copy is still required.
