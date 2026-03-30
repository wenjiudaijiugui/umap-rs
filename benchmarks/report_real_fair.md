# Fair Real-Dataset UMAP Benchmark Report

## Setup
- n_neighbors=15, n_components=2, n_epochs=200, init=random, seed=42
- warmup=1, repeats=5, randomized_order_per_repeat=True
- groups: e2e_default_ann, algo_exact_shared_knn
- thread pinning: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONHASHSEED=0

## Datasets

- breast_cancer: n_used=569, n_original=569, d=30
- digits: n_used=1797, n_original=1797, d=64
- california_housing: n_used=15000, n_original=20640, d=8

## Group A: e2e_default_ann

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.102 ± 0.044 | 345.4 ± 0.2 |
| r_uwot | 0.765 ± 0.045 | 174.8 ± 0.1 |
| rust_umap | 0.129 ± 0.002 | 4.2 ± 0.0 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916196
  - r_uwot: 0.914760
  - rust_umap: 0.914304
- original_knn_recall@15:
  - python_umap_learn: 0.398711
  - r_uwot: 0.407381
  - rust_umap: 0.406327

### Dataset: digits

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 6.541 ± 0.027 | 359.5 ± 0.6 |
| r_uwot | 1.145 ± 0.009 | 182.6 ± 0.1 |
| rust_umap | 0.444 ± 0.005 | 6.7 ± 0.1 |

- sample_size_for_consistency: 1797
- trustworthiness@15:
  - python_umap_learn: 0.967248
  - r_uwot: 0.969163
  - rust_umap: 0.973267
- original_knn_recall@15:
  - python_umap_learn: 0.456279
  - r_uwot: 0.465294
  - rust_umap: 0.471972

### Dataset: california_housing

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 12.524 ± 0.283 | 470.1 ± 0.2 |
| r_uwot | 5.996 ± 0.034 | 268.0 ± 0.1 |
| rust_umap | 3.889 ± 0.099 | 33.8 ± 0.1 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.968255
  - r_uwot: 0.969491
  - rust_umap: 0.966243
- original_knn_recall@15:
  - python_umap_learn: 0.386100
  - r_uwot: 0.393167
  - rust_umap: 0.386833

## Group B: algo_exact_shared_knn

### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.189 ± 0.001 | 309.5 |
| r_uwot | 0.309 ± 0.005 | 188.4 |
| rust_umap | 0.147 ± 0.001 | 4.6 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916197
  - r_uwot: 0.913885
  - rust_umap: 0.920500
- original_knn_recall@15:
  - python_umap_learn: 0.395079
  - r_uwot: 0.399649
  - rust_umap: 0.411482

### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.807 ± 0.001 | 312.0 |
| r_uwot | 0.624 ± 0.007 | 195.4 |
| rust_umap | 0.417 ± 0.000 | 7.7 |

- sample_size_for_consistency: 1797
- trustworthiness@15:
  - python_umap_learn: 0.967185
  - r_uwot: 0.970218
  - rust_umap: 0.971649
- original_knn_recall@15:
  - python_umap_learn: 0.454684
  - r_uwot: 0.466184
  - rust_umap: 0.464144

### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 4.123 ± 0.005 | 333.6 |
| r_uwot | 3.939 ± 0.034 | 265.1 |
| rust_umap | 3.523 ± 0.019 | 39.0 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.967803
  - r_uwot: 0.966221
  - rust_umap: 0.968676
- original_knn_recall@15:
  - python_umap_learn: 0.393533
  - r_uwot: 0.383900
  - rust_umap: 0.393800

## Comparison vs Previous Single-Run report_real.json

### Dataset: breast_cancer

| Implementation | old elapsed (s) | new elapsed mean (s) | new/old elapsed | old RSS (MB) | new RSS mean (MB) | new/old RSS |
|---|---:|---:|---:|---:|---:|---:|
| python_umap_learn | 5.412 | 5.102 | 0.943 | 348.9 | 345.4 | 0.990 |
| r_uwot | 0.788 | 0.765 | 0.970 | 176.6 | 174.8 | 0.989 |
| rust_umap | 0.146 | 0.129 | 0.883 | 4.1 | 4.2 | 1.030 |

### Dataset: digits

| Implementation | old elapsed (s) | new elapsed mean (s) | new/old elapsed | old RSS (MB) | new RSS mean (MB) | new/old RSS |
|---|---:|---:|---:|---:|---:|---:|
| python_umap_learn | 6.667 | 6.541 | 0.981 | 362.1 | 359.5 | 0.993 |
| r_uwot | 1.188 | 1.145 | 0.963 | 184.5 | 182.6 | 0.990 |
| rust_umap | 0.440 | 0.444 | 1.010 | 6.4 | 6.7 | 1.051 |

### Dataset: california_housing

| Implementation | old elapsed (s) | new elapsed mean (s) | new/old elapsed | old RSS (MB) | new RSS mean (MB) | new/old RSS |
|---|---:|---:|---:|---:|---:|---:|
| python_umap_learn | 12.715 | 12.524 | 0.985 | 472.3 | 470.1 | 0.995 |
| r_uwot | 6.026 | 5.996 | 0.995 | 270.1 | 268.0 | 0.992 |
| rust_umap | 3.895 | 3.889 | 0.999 | 33.6 | 33.8 | 1.005 |
