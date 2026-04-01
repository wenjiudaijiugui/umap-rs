from __future__ import annotations

from pathlib import Path
from typing import Any


def configure_rust_bins(
    fair_module: Any,
    *,
    rust_fit_bin: str | None,
    rust_bench_bin: str | None,
    skip_rust_build: bool,
) -> None:
    if skip_rust_build and (not rust_fit_bin or not rust_bench_bin):
        raise ValueError("--skip-rust-build requires --rust-fit-bin and --rust-bench-bin")

    if rust_fit_bin:
        fair_module.RUST_FIT = Path(rust_fit_bin).resolve()
    if rust_bench_bin:
        fair_module.RUST_BENCH = Path(rust_bench_bin).resolve()

    if skip_rust_build:
        if not fair_module.RUST_FIT.exists():
            raise ValueError(f"rust fit binary not found: {fair_module.RUST_FIT}")
        if not fair_module.RUST_BENCH.exists():
            raise ValueError(f"rust bench binary not found: {fair_module.RUST_BENCH}")
        return

    fair_module.build_rust_binaries()
