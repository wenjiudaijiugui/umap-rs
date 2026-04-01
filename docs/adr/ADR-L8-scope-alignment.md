# ADR-L8: Scope Alignment For Documented UMAP Surface

- Status: Accepted
- Date: 2026-04-01
- Owner: L8

## Context

The repository root README and `rust_umap/README.md` had drifted in how they described the currently supported surface. At the same time, the codebase has grown beyond the Python binding MVP: the Rust crate contains additional modules and helpers, while the binding intentionally exposes only a narrower core `Umap` workflow.

This ADR records the L8 decision to converge the documentation first, without treating that convergence as a commitment to implement broader parity immediately.

## Decision

The repository documents the following as the current capability boundary:

1. Stable documented core:
   - Dense single-dataset non-parametric UMAP with `fit`, `fit_transform`, `transform`, and dense-trained `inverse_transform`
   - Precomputed-kNN fit for dense input, with explicit metric alignment
   - Sparse CSR fit MVP in Rust, including dense-query `transform` after sparse training for `euclidean`, `manhattan`, and `cosine`
2. Explicitly unsupported today:
   - `inverse_transform` for sparse-trained models
   - A claim of full `umap-learn` parity
   - A claim of full `pynndescent` ANN parity
3. Python binding alignment contract:
   - The binding aligns to the crate's core single-dataset `Umap` surface
   - The binding does not currently promise parity for crate-only experimental or auxiliary surfaces such as parametric UMAP, aligned UMAP, CLI binaries, or benchmark helpers

## Rationale

- The sparse inverse boundary is enforced in code: sparse-trained models return an error on `inverse_transform`.
- Binding tests cover dense workflows, precomputed-kNN fit, sparse CSR fit / `fit_transform`, and dense-query `transform` after sparse training.
- Treating the Python binding as a core-API MVP avoids over-claiming support for modules that exist in Rust but are not part of the current binding contract.
- Keeping the root README and crate README aligned reduces ambiguity for users deciding what is safe to depend on.

## Consequences

- Future work can expand the surface, but the docs should only be widened when implementation and validation are in place.
- Repository-level messaging should use "scope boundary" language instead of implying blanket parity with every feature available in `umap-learn` or every internal Rust module.
- Any future support for sparse-trained `inverse_transform` or additional binding surfaces should update both README files and this ADR together.
