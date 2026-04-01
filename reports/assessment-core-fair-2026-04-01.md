# Core Fair Benchmark Against umap-learn

- Generated (UTC): 2026-04-01T09:26:08Z
- Python bin: /home/shenshang/paper_reproduction/umap-rs/.venv-bench312/bin/python
- Seed=42, warmup=1, repeats=5, sample_cap=2000
- Datasets: breast_cancer, digits, california_housing
- Quality gate overall_pass: True

## E2E Default Path

| Dataset | Python/Rust Time | Python/Rust RSS | trust delta (rust-py) | recall delta (rust-py) | overlap@15 |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 38.339 | 56.801 | -0.000491 | -0.002929 | 0.573052 |
| digits | 12.911 | 40.162 | -0.000447 | -0.000705 | 0.633945 |
| california_housing | 10.296 | 19.556 | -0.007271 | -0.013133 | 0.488467 |

## Algo Exact Shared kNN

### euclidean

| Dataset | Python/Rust Fit | Python/Rust RSS | trust delta (rust-py) | recall delta (rust-py) | overlap@15 |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 1.362 | 51.735 | -0.001545 | 0.012185 | 0.543995 |
| digits | 1.469 | 31.674 | -0.005191 | 0.000668 | 0.604786 |
| california_housing | 1.439 | 14.279 | -0.001575 | 0.005200 | 0.535567 |

### manhattan

| Dataset | Python/Rust Fit | Python/Rust RSS | trust delta (rust-py) | recall delta (rust-py) | overlap@15 |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 1.218 | 51.653 | 0.004284 | 0.007499 | 0.668658 |
| digits | 1.504 | 31.774 | 0.000587 | 0.003153 | 0.620033 |
| california_housing | 1.449 | 14.228 | -0.001133 | -0.002233 | 0.663033 |

### cosine

| Dataset | Python/Rust Fit | Python/Rust RSS | trust delta (rust-py) | recall delta (rust-py) | overlap@15 |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 1.371 | 51.864 | 0.006359 | -0.003984 | 0.676977 |
| digits | 1.410 | 32.157 | -0.000026 | 0.010091 | 0.636691 |
| california_housing | 1.451 | 14.457 | -0.000298 | 0.006300 | 0.563700 |

## Quality Gate

- e2e_default_ann / breast_cancer / -: pass=True, trust_gap=0.000491, recall_gap=0.002929, min_overlap=0.573052
- e2e_default_ann / digits / -: pass=True, trust_gap=0.000447, recall_gap=0.000705, min_overlap=0.633945
- e2e_default_ann / california_housing / -: pass=True, trust_gap=0.007271, recall_gap=0.013133, min_overlap=0.488467
- algo_exact_shared_knn / breast_cancer / euclidean: pass=True, trust_gap=0.001545, recall_gap=0.012185, min_overlap=0.543995
- algo_exact_shared_knn / digits / euclidean: pass=True, trust_gap=0.005191, recall_gap=0.000668, min_overlap=0.604786
- algo_exact_shared_knn / california_housing / euclidean: pass=True, trust_gap=0.001575, recall_gap=0.005200, min_overlap=0.535567
- algo_exact_shared_knn / breast_cancer / manhattan: pass=True, trust_gap=0.004284, recall_gap=0.007499, min_overlap=0.668658
- algo_exact_shared_knn / digits / manhattan: pass=True, trust_gap=0.000587, recall_gap=0.003153, min_overlap=0.620033
- algo_exact_shared_knn / california_housing / manhattan: pass=True, trust_gap=0.001133, recall_gap=0.002233, min_overlap=0.663033
- algo_exact_shared_knn / breast_cancer / cosine: pass=True, trust_gap=0.006359, recall_gap=0.003984, min_overlap=0.676977
- algo_exact_shared_knn / digits / cosine: pass=True, trust_gap=0.000026, recall_gap=0.010091, min_overlap=0.636691
- algo_exact_shared_knn / california_housing / cosine: pass=True, trust_gap=0.000298, recall_gap=0.006300, min_overlap=0.563700
