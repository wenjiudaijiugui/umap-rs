from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseBins:
    bench_fit_csv: Path
    fit_csv: Path | None = None


def build_release_bins(repo_root: Path, *, need_fit_bin: bool) -> ReleaseBins:
    rust_root = repo_root / "rust_umap"
    command = ["cargo", "build", "--release", "--bin", "bench_fit_csv"]
    if need_fit_bin:
        command.extend(["--bin", "fit_csv"])
    subprocess.run(command, cwd=rust_root, check=True)

    release_dir = rust_root / "target" / "release"
    bench_bin = release_dir / "bench_fit_csv"
    if not bench_bin.exists():
        raise RuntimeError(
            f"expected bench_fit_csv not found under {release_dir}: bench_fit_csv={bench_bin.exists()}"
        )

    fit_bin = release_dir / "fit_csv"
    if need_fit_bin and not fit_bin.exists():
        raise RuntimeError(
            f"expected fit_csv not found under {release_dir}: fit_csv={fit_bin.exists()}"
        )
    return ReleaseBins(bench_fit_csv=bench_bin, fit_csv=fit_bin if need_fit_bin else None)
