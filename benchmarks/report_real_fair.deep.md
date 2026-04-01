# Fair Real-Dataset UMAP Benchmark Report

## Setup
- n_neighbors=15, n_components=2, n_epochs=200, init=random, seed=42
- warmup=1, repeats=5, randomized_order_per_repeat=True
- python_bin=/home/shenshang/miniforge3/envs/umap_bench/bin/python
- rscript_bin=/home/shenshang/miniforge3/envs/umap_bench/bin/Rscript
- groups: e2e_default_ann, algo_exact_shared_knn
- e2e_metric=euclidean
- algo_exact_metrics=euclidean,manhattan,cosine
- thread pinning: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, BLIS_NUM_THREADS=1, NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1, PYTHONHASHSEED=0

## Datasets

- breast_cancer: n_used=569, n_original=569, d=30
- digits: n_used=1797, n_original=1797, d=64
- california_housing: n_used=5000, n_original=20640, d=8

## Group A: e2e_default_ann

### Dataset: breast_cancer

| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |
|---|---:|---:|
| python_umap_learn | 5.133 ± 0.015 | 345.3 ± 0.1 |
| r_uwot | 0.725 ± 0.006 | 174.6 ± 0.1 |
| rust_umap | 0.132 ± 0.003 | 4.4 ± 0.1 |

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
| python_umap_learn | 6.555 ± 0.023 | 358.8 ± 0.1 |
| r_uwot | 1.149 ± 0.015 | 182.6 ± 0.1 |
| rust_umap | 0.443 ± 0.002 | 6.9 ± 0.0 |

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
| python_umap_learn | 10.078 ± 0.093 | 436.9 ± 0.4 |
| r_uwot | 2.535 ± 0.027 | 246.7 ± 0.1 |
| rust_umap | 1.051 ± 0.002 | 15.5 ± 0.0 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.970719
  - r_uwot: 0.968781
  - rust_umap: 0.962680
- original_knn_recall@15:
  - python_umap_learn: 0.378000
  - r_uwot: 0.372767
  - rust_umap: 0.358467

## Group B: algo_exact_shared_knn

### Metric: euclidean

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.188 ± 0.002 | 309.2 |
| r_uwot | 0.365 ± 0.044 | 188.2 |
| rust_umap | 0.121 ± 0.002 | 4.7 |

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
| python_umap_learn | 0.811 ± 0.003 | 312.4 |
| r_uwot | 0.629 ± 0.011 | 194.8 |
| rust_umap | 0.341 ± 0.003 | 7.8 |

- sample_size_for_consistency: 1797
- trustworthiness@15:
  - python_umap_learn: 0.967185
  - r_uwot: 0.970218
  - rust_umap: 0.972141
- original_knn_recall@15:
  - python_umap_learn: 0.454684
  - r_uwot: 0.466184
  - rust_umap: 0.468596

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.385 ± 0.001 | 316.5 |
| r_uwot | 1.441 ± 0.012 | 219.4 |
| rust_umap | 0.950 ± 0.004 | 17.5 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.967939
  - r_uwot: 0.967705
  - rust_umap: 0.971494
- original_knn_recall@15:
  - python_umap_learn: 0.373000
  - r_uwot: 0.374667
  - rust_umap: 0.382800

### Metric: manhattan

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.412 ± 0.002 | 308.8 |
| r_uwot | 0.331 ± 0.018 | 188.1 |
| rust_umap | 0.120 ± 0.000 | 4.6 |

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
| python_umap_learn | 1.297 ± 0.004 | 312.0 |
| r_uwot | 0.619 ± 0.005 | 194.9 |
| rust_umap | 0.335 ± 0.001 | 7.9 |

- sample_size_for_consistency: 1797
- trustworthiness@15:
  - python_umap_learn: 0.957763
  - r_uwot: 0.958546
  - rust_umap: 0.959164
- original_knn_recall@15:
  - python_umap_learn: 0.435726
  - r_uwot: 0.451976
  - rust_umap: 0.461436

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.453 ± 0.009 | 316.4 |
| r_uwot | 1.452 ± 0.006 | 219.5 |
| rust_umap | 0.957 ± 0.002 | 17.7 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.967178
  - r_uwot: 0.965334
  - rust_umap: 0.967275
- original_knn_recall@15:
  - python_umap_learn: 0.371000
  - r_uwot: 0.363500
  - rust_umap: 0.369133

### Metric: cosine

#### Dataset: breast_cancer

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 0.161 ± 0.001 | 309.8 |
| r_uwot | 0.309 ± 0.005 | 188.4 |
| rust_umap | 0.118 ± 0.000 | 4.6 |

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
| python_umap_learn | 0.490 ± 0.008 | 311.6 |
| r_uwot | 0.613 ± 0.004 | 195.0 |
| rust_umap | 0.330 ± 0.000 | 7.8 |

- sample_size_for_consistency: 1797
- trustworthiness@15:
  - python_umap_learn: 0.973401
  - r_uwot: 0.973061
  - rust_umap: 0.972564
- original_knn_recall@15:
  - python_umap_learn: 0.485142
  - r_uwot: 0.481247
  - rust_umap: 0.487219

#### Dataset: california_housing

| Implementation | Fit mean±std (s) | Process max RSS (MB) |
|---|---:|---:|
| python_umap_learn | 1.346 ± 0.008 | 316.6 |
| r_uwot | 1.427 ± 0.017 | 219.5 |
| rust_umap | 0.937 ± 0.008 | 17.5 |

- sample_size_for_consistency: 2000
- trustworthiness@15:
  - python_umap_learn: 0.958510
  - r_uwot: 0.958564
  - rust_umap: 0.958566
- original_knn_recall@15:
  - python_umap_learn: 0.330000
  - r_uwot: 0.326467
  - rust_umap: 0.333633

## Quality Gate Verdict

- overall_pass: PASS
- run policy: warmup>=1, repeats>=5 (observed warmup=1, repeats=5)
- consistency policy: trust_gap<=0.030000, recall_gap<=0.080000, min_pairwise_overlap@15>=0.350000

| Group | Dataset | Metric | trust_gap | recall_gap | min_pairwise_overlap@15 | Pass |
|---|---|---|---:|---:|---:|---:|
| e2e_default_ann | breast_cancer | - | 0.001892 | 0.008670 | 0.511189 | yes |
| e2e_default_ann | digits | - | 0.006019 | 0.015693 | 0.541495 | yes |
| e2e_default_ann | california_housing | - | 0.008039 | 0.019533 | 0.487533 | yes |
| algo_exact_shared_knn | breast_cancer | euclidean | 0.008633 | 0.017223 | 0.480844 | yes |
| algo_exact_shared_knn | digits | euclidean | 0.004955 | 0.013912 | 0.541421 | yes |
| algo_exact_shared_knn | california_housing | euclidean | 0.003789 | 0.009800 | 0.572600 | yes |
| algo_exact_shared_knn | breast_cancer | manhattan | 0.018881 | 0.020621 | 0.489982 | yes |
| algo_exact_shared_knn | digits | manhattan | 0.001402 | 0.025710 | 0.524875 | yes |
| algo_exact_shared_knn | california_housing | manhattan | 0.001941 | 0.007500 | 0.628667 | yes |
| algo_exact_shared_knn | breast_cancer | cosine | 0.005625 | 0.003515 | 0.638547 | yes |
| algo_exact_shared_knn | digits | cosine | 0.000837 | 0.005973 | 0.636802 | yes |
| algo_exact_shared_knn | california_housing | cosine | 0.000056 | 0.007167 | 0.606533 | yes |

- violations: none
