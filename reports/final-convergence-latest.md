# Final Convergence Report (L7) - 2026-04-01

- Generated (UTC): 2026-04-01T06:36:13Z
- Decision: **GO**
- Status: `completed`

## 1) Gate Summary

| Gate | Pass | Return Code | Artifact |
|---|---:|---:|---|
| `wave1_smoke` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/wave1-smoke.json` |
| `ann_e2e_smoke` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/ann-e2e-smoke.json` |
| `consistency_smoke` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/consistency-smoke.json` |
| `no_regression_smoke:euclidean` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/no-regression-smoke-euclidean.json` |
| `no_regression_smoke:manhattan` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/no-regression-smoke-manhattan.json` |
| `no_regression_smoke:cosine` | true | 0 | `benchmarks/release-prep-regression.final.artifacts/no-regression-smoke-cosine.json` |

## 2) Key Inputs (L1/L2/L3/L6)

- L1 deep benchmark: `quality_gate.overall_pass=True`; warmup/repeats = 1/5 (`benchmarks/report_real_fair.deep.json`).
- L2 ANN coverage: `california_housing.knn_strategy.equivalence=not_equivalent` (`benchmarks/report_ecosystem_python_binding.ann.json`).
- L3 CI quality hardening evidence: `.github/workflows/ci.yml` includes `rust-fmt` + `rust-verify(clippy -D warnings)` and smoke jobs depend on `rust-verify`.
- L6 evidence governance: branch reports validated and documented in `reports/algo-repro-wave/*` and `reports/TEMP_parallel_plan_2026-03-31.md#evidence-governance-revision-2026-04-01`.

## 3) Blocking Issue

- None. release-prep gates all passed in this run.

## 4) No-Regression Snapshot

- `cosine`: fit ratio median/p75=1.003726/1.005562, rss ratio median/p75=0.993929/1.004067.
- `euclidean`: fit ratio median/p75=0.999600/1.006601, rss ratio median/p75=0.985682/0.993946.
- `manhattan`: fit ratio median/p75=1.003685/1.007770, rss ratio median/p75=0.998005/1.002155.

## 5) Next Actions

1. 将 L7 final-convergence 报告作为当前发布判定快照。
2. 后续若引入代码/基准改动，重新执行 release_prep_regression 并更新报告。
