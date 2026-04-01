# Fair Real-Dataset UMAP Benchmark Report

## Setup
- n_neighbors=15, n_components=2, n_epochs=200, init=random, seed=42
- warmup=0, repeats=1, randomized_order_per_repeat=True
- python_bin=python
- rscript_bin=Rscript
- groups: e2e_default_ann, algo_exact_shared_knn
- e2e_metric=euclidean
- algo_exact_metrics=euclidean,manhattan,cosine
- thread pinning: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONHASHSEED=0

## Datasets

- breast_cancer: n_used=569, n_original=569, d=30
- digits: n_used=1797, n_original=1797, d=64
- california_housing: n_used=4000, n_original=20640, d=8

## Group A: e2e_default_ann

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.123 ± 0.000 | 345.4 ± 0.0 |
| r_uwot | 0.834 ± 0.000 | 174.6 ± 0.0 |
| rust_umap | 0.136 ± 0.000 | 4.4 ± 0.0 |

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
| python_umap_learn | 6.504 ± 0.000 | 360.2 ± 0.0 |
| r_uwot | 1.132 ± 0.000 | 182.6 ± 0.0 |
| rust_umap | 0.443 ± 0.000 | 6.9 ± 0.0 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.956440
  - r_uwot: 0.963531
  - rust_umap: 0.967299
- original_knn_recall@15:
  - python_umap_learn: 0.458000
  - r_uwot: 0.474400
  - rust_umap: 0.474067

### Dataset: california_housing

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 11.367 ± 0.000 | 409.4 ± 0.0 |
| r_uwot | 1.461 ± 0.000 | 195.8 ± 0.0 |
| rust_umap | 1.065 ± 0.000 | 12.0 ± 0.0 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.959369
  - r_uwot: 0.964970
  - rust_umap: 0.966473
- original_knn_recall@15:
  - python_umap_learn: 0.423267
  - r_uwot: 0.434267
  - rust_umap: 0.434200

## Group B: algo_exact_shared_knn

### Metric: euclidean

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.303 ± 0.000 | 309.5 |
| r_uwot | 0.341 ± 0.000 | 188.4 |
| rust_umap | 0.119 ± 0.000 | 4.4 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.916197
  - r_uwot: 0.913885
  - rust_umap: 0.922518
- original_knn_recall@15:
  - python_umap_learn: 0.395079
  - r_uwot: 0.399649
  - rust_umap: 0.412302

#### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.921 ± 0.000 | 312.1 |
| r_uwot | 0.656 ± 0.000 | 194.7 |
| rust_umap | 0.339 ± 0.000 | 7.0 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.957234
  - r_uwot: 0.960807
  - rust_umap: 0.965704
- original_knn_recall@15:
  - python_umap_learn: 0.462200
  - r_uwot: 0.471533
  - rust_umap: 0.474867

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 2.235 ± 0.000 | 314.6 |
| r_uwot | 1.215 ± 0.000 | 210.5 |
| rust_umap | 0.752 ± 0.000 | 14.6 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.962496
  - r_uwot: 0.966558
  - rust_umap: 0.966600
- original_knn_recall@15:
  - python_umap_learn: 0.432800
  - r_uwot: 0.448933
  - rust_umap: 0.446133

### Metric: manhattan

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.521 ± 0.000 | 309.7 |
| r_uwot | 0.330 ± 0.000 | 188.1 |
| rust_umap | 0.119 ± 0.000 | 4.4 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.885940
  - r_uwot: 0.899416
  - rust_umap: 0.904821
- original_knn_recall@15:
  - python_umap_learn: 0.387698
  - r_uwot: 0.403281
  - rust_umap: 0.408319

#### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 2.415 ± 0.000 | 312.2 |
| r_uwot | 0.656 ± 0.000 | 195.1 |
| rust_umap | 0.340 ± 0.000 | 7.0 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.952444
  - r_uwot: 0.951816
  - rust_umap: 0.954473
- original_knn_recall@15:
  - python_umap_learn: 0.444133
  - r_uwot: 0.448733
  - rust_umap: 0.460533

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 2.294 ± 0.000 | 314.5 |
| r_uwot | 1.231 ± 0.000 | 210.6 |
| rust_umap | 0.741 ± 0.000 | 14.5 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.957255
  - r_uwot: 0.956148
  - rust_umap: 0.956414
- original_knn_recall@15:
  - python_umap_learn: 0.423533
  - r_uwot: 0.414000
  - rust_umap: 0.421400

### Metric: cosine

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.279 ± 0.000 | 309.1 |
| r_uwot | 0.336 ± 0.000 | 188.4 |
| rust_umap | 0.117 ± 0.000 | 4.4 |

- sample_size_for_consistency: 569
- trustworthiness@15:
  - python_umap_learn: 0.893242
  - r_uwot: 0.898868
  - rust_umap: 0.897254
- original_knn_recall@15:
  - python_umap_learn: 0.435501
  - r_uwot: 0.437610
  - rust_umap: 0.439016

#### Dataset: digits

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.616 ± 0.000 | 311.5 |
| r_uwot | 0.653 ± 0.000 | 195.3 |
| rust_umap | 0.346 ± 0.000 | 7.1 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.966625
  - r_uwot: 0.965524
  - rust_umap: 0.968654
- original_knn_recall@15:
  - python_umap_learn: 0.480067
  - r_uwot: 0.475733
  - rust_umap: 0.475467

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 2.247 ± 0.000 | 314.6 |
| r_uwot | 1.213 ± 0.000 | 210.8 |
| rust_umap | 0.718 ± 0.000 | 14.4 |

- sample_size_for_consistency: 1000
- trustworthiness@15:
  - python_umap_learn: 0.955124
  - r_uwot: 0.952817
  - rust_umap: 0.952803
- original_knn_recall@15:
  - python_umap_learn: 0.401667
  - r_uwot: 0.390933
  - rust_umap: 0.403200
