#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import procrustes
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from reporting_utils import display_executable


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"

DATA_DIR = BENCH_DIR / "data_real_ecosystem"
KNN_DIR = BENCH_DIR / "knn_real_ecosystem"
OUT_DIR = BENCH_DIR / "outputs_real_ecosystem"
TIME_DIR = BENCH_DIR / "time_real_ecosystem"

REPORT_JSON = BENCH_DIR / "report_ecosystem_python_binding.json"
REPORT_MD = BENCH_DIR / "report_ecosystem_python_binding.md"

GROUP_E2E_MIXED = "e2e_mixed_knn_strategy"
GROUP_ALGO_EXACT = "algo_exact_shared_knn_exact"

RUN_PY_E2E = BENCH_DIR / "run_python_umap.py"
RUN_PY_ALGO = BENCH_DIR / "run_python_umap_algo.py"
RUN_RUST_PY_E2E = BENCH_DIR / "run_rust_umap_py.py"
RUN_RUST_PY_ALGO = BENCH_DIR / "run_rust_umap_py_algo.py"

IMPLS = ["python_umap_learn", "rust_umap_py"]

N_NEIGHBORS = 15
N_COMPONENTS = 2
N_EPOCHS = 200
INIT = "random"
METRIC = "euclidean"
E2E_RUST_APPROX_THRESHOLD = 4096

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

_GNU_TIME_BIN_CACHE: Optional[str] = None


@dataclass
class DatasetSpec:
    name: str
    x: np.ndarray
    original_samples: int
    used_samples: int
    n_features: int


@dataclass
class TimedRun:
    elapsed_sec: float
    max_rss_mb: float
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark rust_umap_py vs python umap-learn with fair settings")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sample-cap-consistency", type=int, default=2000)
    p.add_argument("--include-california", action="store_true")
    p.add_argument(
        "--ensure-ann-coverage",
        action="store_true",
        help=(
            "append a large enough california_housing slice when needed so the mixed e2e group "
            "covers rust_umap_py approximate ANN at least once"
        ),
    )
    p.add_argument("--large-max-samples", type=int, default=15000)
    p.add_argument("--report-json", default=str(REPORT_JSON))
    p.add_argument("--report-md", default=str(REPORT_MD))
    return p.parse_args()


def ensure_dirs() -> None:
    for d in [DATA_DIR, KNN_DIR, OUT_DIR, TIME_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_dataset_csv(path: Path, x: np.ndarray) -> None:
    np.savetxt(path, x, delimiter=",", fmt="%.8f")


def parse_max_rss_mb(time_file: Path) -> float:
    txt = time_file.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)
    if not m:
        return float("nan")
    return float(m.group(1)) / 1024.0


def _unique_non_empty(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _time_candidates() -> List[str]:
    override = os.environ.get("UMAP_BENCH_TIME_BIN", "").strip()
    candidates: List[str] = []
    if override:
        candidates.append(override)

    for exe in ["gtime", "time"]:
        resolved = shutil.which(exe)
        if resolved:
            candidates.append(resolved)

    candidates.extend(["/usr/bin/time", "gtime", "time"])
    return _unique_non_empty(candidates)


def _probe_gnu_time(candidate: str) -> Tuple[bool, str]:
    cmd = [candidate, "-v", "true"]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return False, "not found"
    except PermissionError:
        return False, "not executable"
    except OSError as exc:
        return False, f"probe failed: {exc}"

    out = f"{proc.stdout}\n{proc.stderr}"
    if "Maximum resident set size" in out:
        return True, "ok"

    version_hint = ""
    try:
        ver = subprocess.run(
            [candidate, "--version"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        ver_out = f"{ver.stdout}\n{ver.stderr}"
        first_line = next((line.strip() for line in ver_out.splitlines() if line.strip()), "")
        if first_line:
            version_hint = f" (version: {first_line[:80]})"
    except OSError:
        pass

    return False, f"missing GNU max RSS field in '-v' output{version_hint}"


def resolve_time_bin() -> str:
    global _GNU_TIME_BIN_CACHE
    if _GNU_TIME_BIN_CACHE:
        return _GNU_TIME_BIN_CACHE

    attempted: List[str] = []
    for candidate in _time_candidates():
        ok, reason = _probe_gnu_time(candidate)
        attempted.append(f"{candidate}: {reason}")
        if ok:
            _GNU_TIME_BIN_CACHE = candidate
            return candidate

    attempted_str = "; ".join(attempted) if attempted else "none"
    raise RuntimeError(
        "Unable to find GNU time with '-v' max RSS reporting. "
        f"Tried: {attempted_str}. "
        "Set UMAP_BENCH_TIME_BIN to a GNU time executable path. "
        "Install hint: Debian/Ubuntu `apt-get install time`; macOS `brew install gnu-time`."
    )


def run_timed(cmd: List[str], time_file: Path) -> TimedRun:
    wrapped = [resolve_time_bin(), "-v", "-o", str(time_file)] + cmd
    env = os.environ.copy()
    env.update(THREAD_ENV)

    t0 = time.perf_counter()
    proc = subprocess.run(wrapped, check=True, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    max_rss_mb = parse_max_rss_mb(time_file)
    return TimedRun(elapsed_sec=elapsed, max_rss_mb=max_rss_mb, stdout=proc.stdout, stderr=proc.stderr)


def parse_json_line(text: str) -> Dict[str, object]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("No JSON payload found in command output")


def load_california_housing_dataset(
    seed: int,
    requested_max_samples: int,
    minimum_required_samples: int = 0,
) -> DatasetSpec:
    x_large = fetch_california_housing().data.astype(np.float32)
    original_n = int(x_large.shape[0])

    target_samples = original_n
    if requested_max_samples > 0:
        target_samples = min(original_n, requested_max_samples)
    if minimum_required_samples > 0:
        target_samples = min(original_n, max(target_samples, minimum_required_samples))

    if original_n > target_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(original_n, size=target_samples, replace=False)
        x_large = x_large[idx]

    used_n = int(x_large.shape[0])
    x_large = StandardScaler().fit_transform(x_large).astype(np.float32)
    return DatasetSpec(
        name="california_housing",
        x=x_large,
        original_samples=original_n,
        used_samples=used_n,
        n_features=int(x_large.shape[1]),
    )


def load_datasets(
    seed: int,
    include_california: bool,
    large_max_samples: int,
    ensure_ann_coverage: bool,
) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []

    x_bc = load_breast_cancer().data.astype(np.float32)
    x_bc = StandardScaler().fit_transform(x_bc).astype(np.float32)
    specs.append(
        DatasetSpec(
            name="breast_cancer",
            x=x_bc,
            original_samples=int(x_bc.shape[0]),
            used_samples=int(x_bc.shape[0]),
            n_features=int(x_bc.shape[1]),
        )
    )

    x_digits = load_digits().data.astype(np.float32)
    x_digits = StandardScaler().fit_transform(x_digits).astype(np.float32)
    specs.append(
        DatasetSpec(
            name="digits",
            x=x_digits,
            original_samples=int(x_digits.shape[0]),
            used_samples=int(x_digits.shape[0]),
            n_features=int(x_digits.shape[1]),
        )
    )

    need_ann_dataset = ensure_ann_coverage and all(
        spec.used_samples <= E2E_RUST_APPROX_THRESHOLD for spec in specs
    )
    if include_california or need_ann_dataset:
        min_required_samples = E2E_RUST_APPROX_THRESHOLD + 1 if ensure_ann_coverage else 0
        specs.append(
            load_california_housing_dataset(
                seed=seed,
                requested_max_samples=large_max_samples,
                minimum_required_samples=min_required_samples,
            )
        )

    return specs


def compute_shared_exact_knn(x: np.ndarray, k: int, idx_path: Path, dist_path: Path) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean", n_jobs=1)
    nbrs.fit(x)
    dists, idx = nbrs.kneighbors(x)
    idx = idx[:, 1 : k + 1].astype(np.int64)
    dists = dists[:, 1 : k + 1].astype(np.float32)

    np.savetxt(idx_path, idx, delimiter=",", fmt="%d")
    np.savetxt(dist_path, dists, delimiter=",", fmt="%.8f")
    return idx


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
    }


def knn_indices(emb: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(emb)
    idx = nn.kneighbors(return_distance=False)
    return idx[:, 1:]


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    idx_a = knn_indices(a, k)
    idx_b = knn_indices(b, k)
    n = idx_a.shape[0]
    overlap = 0.0
    for i in range(n):
        sa = set(idx_a[i].tolist())
        sb = set(idx_b[i].tolist())
        overlap += len(sa & sb) / float(k)
    return overlap / n


def knn_recall_vs_original(orig_knn_idx: np.ndarray, emb: np.ndarray, k: int) -> float:
    emb_idx = knn_indices(emb, k)
    n = orig_knn_idx.shape[0]
    recall = 0.0
    for i in range(n):
        so = set(orig_knn_idx[i].tolist())
        se = set(emb_idx[i].tolist())
        recall += len(so & se) / float(k)
    return recall / n


def compute_consistency(
    x: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    seed: int,
    orig_knn_idx: np.ndarray,
    k: int,
    sample_cap: int,
) -> Dict[str, object]:
    n = x.shape[0]
    if n > sample_cap:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n, size=sample_cap, replace=False)
    else:
        sample_idx = np.arange(n)

    x_s = x[sample_idx]
    emb_s = {name: emb[sample_idx] for name, emb in embeddings.items()}
    if n > sample_cap:
        orig_s = knn_indices(x_s, k)
    else:
        orig_s = orig_knn_idx[sample_idx]

    trust = {
        name: float(trustworthiness(x_s, emb, n_neighbors=k))
        for name, emb in emb_s.items()
    }
    recall = {
        name: float(knn_recall_vs_original(orig_s, emb, k))
        for name, emb in emb_s.items()
    }

    py_name = "python_umap_learn"
    rust_name = "rust_umap_py"
    _, _, disparity = procrustes(emb_s[py_name], emb_s[rust_name])
    overlap = knn_overlap(emb_s[py_name], emb_s[rust_name], k)

    pairwise = {
        f"{py_name}__vs__{rust_name}": {
            "procrustes_disparity": float(disparity),
            "knn_overlap_at_15": float(overlap),
        }
    }

    return {
        "sample_size_for_consistency": int(len(sample_idx)),
        "trustworthiness_at_15": trust,
        "original_knn_recall_at_15": recall,
        "pairwise": pairwise,
    }


def e2e_cmd(python_bin: Path, impl: str, data_path: Path, out_path: Path, seed: int) -> List[str]:
    if impl == "python_umap_learn":
        return [
            str(python_bin),
            "-I",
            str(RUN_PY_E2E),
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--n-neighbors",
            str(N_NEIGHBORS),
            "--n-components",
            str(N_COMPONENTS),
            "--n-epochs",
            str(N_EPOCHS),
            "--seed",
            str(seed),
            "--init",
            INIT,
        ]
    if impl == "rust_umap_py":
        return [
            str(python_bin),
            "-I",
            str(RUN_RUST_PY_E2E),
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--n-neighbors",
            str(N_NEIGHBORS),
            "--n-components",
            str(N_COMPONENTS),
            "--n-epochs",
            str(N_EPOCHS),
            "--seed",
            str(seed),
            "--init",
            INIT,
            "--metric",
            METRIC,
            "--use-approximate-knn",
            "true",
            "--approx-knn-candidates",
            "30",
            "--approx-knn-iters",
            "10",
            "--approx-knn-threshold",
            str(E2E_RUST_APPROX_THRESHOLD),
        ]
    raise ValueError(f"unknown impl: {impl}")


def algo_cmd(
    python_bin: Path,
    impl: str,
    data_path: Path,
    out_path: Path,
    seed: int,
    warmup: int,
    repeats: int,
    idx_path: Path,
    dist_path: Path,
) -> List[str]:
    if impl == "python_umap_learn":
        return [
            str(python_bin),
            "-I",
            str(RUN_PY_ALGO),
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--n-neighbors",
            str(N_NEIGHBORS),
            "--n-components",
            str(N_COMPONENTS),
            "--n-epochs",
            str(N_EPOCHS),
            "--seed",
            str(seed),
            "--init",
            INIT,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--knn-indices",
            str(idx_path),
            "--knn-dists",
            str(dist_path),
        ]

    if impl == "rust_umap_py":
        return [
            str(python_bin),
            "-I",
            str(RUN_RUST_PY_ALGO),
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--n-neighbors",
            str(N_NEIGHBORS),
            "--n-components",
            str(N_COMPONENTS),
            "--n-epochs",
            str(N_EPOCHS),
            "--seed",
            str(seed),
            "--init",
            INIT,
            "--metric",
            METRIC,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--knn-indices",
            str(idx_path),
            "--knn-dists",
            str(dist_path),
            "--knn-metric",
            METRIC,
            "--use-approximate-knn",
            "false",
        ]

    raise ValueError(f"unknown impl: {impl}")


def benchmark_e2e(
    python_bin: Path,
    spec: DatasetSpec,
    data_path: Path,
    seed: int,
    warmup: int,
    repeats: int,
    sample_cap_consistency: int,
    orig_knn_idx: np.ndarray,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed + spec.used_samples + 11)

    raw: Dict[str, Dict[str, List[float]]] = {
        impl: {"elapsed_sec": [], "max_rss_mb": []} for impl in IMPLS
    }
    final_embedding_paths: Dict[str, Path] = {}

    total = warmup + repeats
    for rep in range(total):
        order = [str(x) for x in rng.permutation(IMPLS)]
        print(f"[ecosystem][e2e][{spec.name}] rep={rep + 1}/{total} order={order}")
        for impl in order:
            out_path = OUT_DIR / f"{spec.name}__{impl}__e2e__rep{rep}.csv"
            time_path = TIME_DIR / f"{spec.name}__{impl}__e2e__rep{rep}.time.txt"
            run = run_timed(e2e_cmd(python_bin, impl, data_path, out_path, seed), time_path)
            if rep >= warmup:
                raw[impl]["elapsed_sec"].append(run.elapsed_sec)
                raw[impl]["max_rss_mb"].append(run.max_rss_mb)
                final_embedding_paths[impl] = out_path

    summary = {}
    for impl in IMPLS:
        elapsed_stat = summarize(raw[impl]["elapsed_sec"])
        rss_stat = summarize(raw[impl]["max_rss_mb"])
        summary[impl] = {
            "elapsed_mean_sec": elapsed_stat["mean"],
            "elapsed_std_sec": elapsed_stat["std"],
            "max_rss_mean_mb": rss_stat["mean"],
            "max_rss_std_mb": rss_stat["std"],
            "elapsed_samples_sec": raw[impl]["elapsed_sec"],
            "max_rss_samples_mb": raw[impl]["max_rss_mb"],
        }

    embeddings = {
        impl: np.loadtxt(path, delimiter=",", dtype=np.float32)
        for impl, path in final_embedding_paths.items()
    }
    for impl, emb in list(embeddings.items()):
        if emb.ndim == 1:
            embeddings[impl] = emb.reshape(-1, N_COMPONENTS)

    consistency = compute_consistency(
        x=spec.x,
        embeddings=embeddings,
        seed=seed + spec.used_samples,
        orig_knn_idx=orig_knn_idx,
        k=N_NEIGHBORS,
        sample_cap=sample_cap_consistency,
    )

    rust_runtime_strategy = "approximate_ann" if spec.used_samples > E2E_RUST_APPROX_THRESHOLD else "exact"
    strategy_equivalence = "strict_exact" if rust_runtime_strategy == "exact" else "not_equivalent"
    strategy_note = (
        "Both implementations used exact kNN for this dataset."
        if strategy_equivalence == "strict_exact"
        else (
            "Python path stays exact (force_approximation_algorithm=False), "
            "while rust_umap_py switched to ANN because n_samples exceeded approx_knn_threshold."
        )
    )

    return {
        "speed_and_memory": summary,
        "consistency": consistency,
        "knn_strategy": {
            "equivalence": strategy_equivalence,
            "note": strategy_note,
            "python_umap_learn": {
                "strategy": "exact",
                "force_approximation_algorithm": False,
            },
            "rust_umap_py": {
                "strategy": rust_runtime_strategy,
                "use_approximate_knn": True,
                "approx_knn_threshold": E2E_RUST_APPROX_THRESHOLD,
            },
        },
    }


def benchmark_algo_exact(
    python_bin: Path,
    spec: DatasetSpec,
    data_path: Path,
    seed: int,
    warmup: int,
    repeats: int,
    sample_cap_consistency: int,
    orig_knn_idx: np.ndarray,
    idx_path: Path,
    dist_path: Path,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed + spec.used_samples + 22)
    order = [str(x) for x in rng.permutation(IMPLS)]
    print(f"[ecosystem][algo][{spec.name}] invocation_order={order}")

    summary = {}
    final_embedding_paths: Dict[str, Path] = {}

    for impl in order:
        out_path = OUT_DIR / f"{spec.name}__{impl}__algo_exact.csv"
        time_path = TIME_DIR / f"{spec.name}__{impl}__algo_exact.time.txt"
        cmd = algo_cmd(python_bin, impl, data_path, out_path, seed, warmup, repeats, idx_path, dist_path)
        run = run_timed(cmd, time_path)
        payload = parse_json_line(run.stdout)
        summary[impl] = {
            "fit_mean_sec": float(payload.get("fit_mean_sec", float("nan"))),
            "fit_std_sec": float(payload.get("fit_std_sec", float("nan"))),
            "fit_times_sec": payload.get("fit_times_sec", []),
            "process_elapsed_sec": run.elapsed_sec,
            "process_max_rss_mb": run.max_rss_mb,
            "interop": payload.get("interop", {}),
        }
        final_embedding_paths[impl] = out_path

    summary = {impl: summary[impl] for impl in IMPLS}

    embeddings = {
        impl: np.loadtxt(path, delimiter=",", dtype=np.float32)
        for impl, path in final_embedding_paths.items()
    }
    for impl, emb in list(embeddings.items()):
        if emb.ndim == 1:
            embeddings[impl] = emb.reshape(-1, N_COMPONENTS)

    consistency = compute_consistency(
        x=spec.x,
        embeddings=embeddings,
        seed=seed + spec.used_samples,
        orig_knn_idx=orig_knn_idx,
        k=N_NEIGHBORS,
        sample_cap=sample_cap_consistency,
    )

    return {
        "fit_timing_and_process_memory": summary,
        "consistency": consistency,
    }


def render_markdown(report: Dict[str, object], python_bin: Path) -> str:
    lines: List[str] = []
    lines.append("# Ecosystem Python Binding Benchmark Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"- n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS}, n_epochs={N_EPOCHS}, init={INIT}, metric={METRIC}, seed={report['config']['seed']}"
    )
    lines.append(
        f"- warmup={report['config']['warmup']}, repeats={report['config']['repeats']}, python_bin={display_executable(python_bin)}"
    )
    lines.append("- implementations: python_umap_learn, rust_umap_py")
    lines.append("- groups: e2e_mixed_knn_strategy, algo_exact_shared_knn_exact")
    lines.append("- thread pinning: " + ", ".join([f"{k}={v}" for k, v in THREAD_ENV.items()]))
    lines.append("")

    lines.append("## Datasets")
    lines.append("")
    for ds in report["config"]["datasets"]:
        lines.append(
            f"- {ds['name']}: n_used={ds['n_samples_used']}, n_original={ds['n_samples_original']}, d={ds['n_features']}"
        )
    lines.append("")

    lines.append("## Group A: e2e_mixed_knn_strategy")
    lines.append("")
    lines.append(
        "- This group intentionally keeps each implementation's runtime defaults; strategy may differ across implementations/datasets."
    )
    lines.append(
        "- python_umap_learn: exact kNN (`force_approximation_algorithm=False`); rust_umap_py: adaptive (`use_approximate_knn=True`, threshold=4096)."
    )
    lines.append("")
    for ds_name, ds in report["groups"][GROUP_E2E_MIXED].items():
        lines.append(f"### Dataset: {ds_name}")
        lines.append("")
        lines.append("| Implementation | Elapsed mean±std (s) | Max RSS mean±std (MB) |")
        lines.append("|---|---:|---:|")
        for impl in IMPLS:
            m = ds["speed_and_memory"][impl]
            lines.append(
                f"| {impl} | {m['elapsed_mean_sec']:.3f} ± {m['elapsed_std_sec']:.3f} | {m['max_rss_mean_mb']:.1f} ± {m['max_rss_std_mb']:.1f} |"
            )

        c = ds["consistency"]
        s = ds["knn_strategy"]
        lines.append("")
        lines.append(f"- knn_strategy_equivalence: {s['equivalence']}")
        lines.append(f"- knn_strategy_note: {s['note']}")
        lines.append(f"- python_umap_learn strategy: {s['python_umap_learn']['strategy']}")
        lines.append(f"- rust_umap_py strategy: {s['rust_umap_py']['strategy']}")
        lines.append(f"- sample_size_for_consistency: {c['sample_size_for_consistency']}")
        lines.append("- trustworthiness@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['trustworthiness_at_15'][impl]:.6f}")
        lines.append("- original_knn_recall@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['original_knn_recall_at_15'][impl]:.6f}")
        pair = c["pairwise"]["python_umap_learn__vs__rust_umap_py"]
        lines.append(
            f"- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity={pair['procrustes_disparity']:.6f}, knn_overlap@15={pair['knn_overlap_at_15']:.6f}"
        )
        lines.append("")

    lines.append("## Group B: algo_exact_shared_knn_exact")
    lines.append("")
    lines.append("- This group enforces strict comparability: both implementations use the same precomputed exact shared kNN graph.")
    lines.append("")
    for ds_name, ds in report["groups"][GROUP_ALGO_EXACT].items():
        lines.append(f"### Dataset: {ds_name}")
        lines.append("")
        lines.append("| Implementation | Fit mean±std (s) | Process max RSS (MB) |")
        lines.append("|---|---:|---:|")
        for impl in IMPLS:
            m = ds["fit_timing_and_process_memory"][impl]
            lines.append(
                f"| {impl} | {m['fit_mean_sec']:.3f} ± {m['fit_std_sec']:.3f} | {m['process_max_rss_mb']:.1f} |"
            )

        c = ds["consistency"]
        lines.append("")
        lines.append(f"- sample_size_for_consistency: {c['sample_size_for_consistency']}")
        lines.append("- trustworthiness@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['trustworthiness_at_15'][impl]:.6f}")
        lines.append("- original_knn_recall@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['original_knn_recall_at_15'][impl]:.6f}")
        pair = c["pairwise"]["python_umap_learn__vs__rust_umap_py"]
        lines.append(
            f"- pairwise python_umap_learn vs rust_umap_py: procrustes_disparity={pair['procrustes_disparity']:.6f}, knn_overlap@15={pair['knn_overlap_at_15']:.6f}"
        )
        lines.append("")

    lines.append("## Interop Audit")
    lines.append("")
    for item in report["interop_audit"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    python_bin = Path(args.python_bin)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    ensure_dirs()

    datasets = load_datasets(
        seed=args.seed,
        include_california=args.include_california,
        large_max_samples=args.large_max_samples,
        ensure_ann_coverage=args.ensure_ann_coverage,
    )

    report: Dict[str, object] = {
        "config": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dataset_type": "real",
            "n_neighbors": N_NEIGHBORS,
            "n_components": N_COMPONENTS,
            "n_epochs": N_EPOCHS,
            "init": INIT,
            "metric": METRIC,
            "ensure_ann_coverage": args.ensure_ann_coverage,
            "datasets": [],
            "thread_env": THREAD_ENV,
        },
        "groups": {
            GROUP_E2E_MIXED: {},
            GROUP_ALGO_EXACT: {},
        },
        "group_definitions": {
            GROUP_E2E_MIXED: {
                "fairness": "not_strict_by_design",
                "python_umap_learn_knn": "exact (force_approximation_algorithm=False)",
                "rust_umap_py_knn": (
                    "adaptive (use_approximate_knn=True, approx_knn_threshold="
                    f"{E2E_RUST_APPROX_THRESHOLD})"
                ),
                "note": (
                    "This group is for ecosystem-default behavior, not strict ANN/exact parity. "
                    "Dataset-level runtime strategy is reported per implementation."
                ),
            },
            GROUP_ALGO_EXACT: {
                "fairness": "strict_exact_equivalent",
                "python_umap_learn_knn": "shared precomputed exact kNN",
                "rust_umap_py_knn": "shared precomputed exact kNN",
                "note": "Primary fairness group for algorithm-level speed/memory/consistency conclusions.",
            },
        },
        "interop_audit": [
            "Input dtype is normalized to float32 before crossing Python/Rust boundary.",
            "kNN indices use int64 and kNN distances use float32 in the binding path.",
            "Thread counts are pinned to 1 for BLAS/OpenMP/Numba to avoid cross-runtime thread bias.",
            "Random seed is aligned across implementations with seed=42 by default.",
            "Algorithm timing scope in both bindings is aligned to the fit_transform* call only.",
            "No post-fit dtype conversion/copy is included in algorithm timers on either side.",
            "Current rust_umap core stores row-major Vec<Vec<f32>>, so one boundary copy is still required.",
        ],
    }

    for spec in datasets:
        data_path = DATA_DIR / f"{spec.name}.csv"
        idx_path = KNN_DIR / f"{spec.name}__knn_idx.csv"
        dist_path = KNN_DIR / f"{spec.name}__knn_dist.csv"

        save_dataset_csv(data_path, spec.x)
        orig_knn_idx = compute_shared_exact_knn(spec.x, N_NEIGHBORS, idx_path, dist_path)

        report["config"]["datasets"].append(
            {
                "name": spec.name,
                "n_samples_original": spec.original_samples,
                "n_samples_used": spec.used_samples,
                "n_features": spec.n_features,
            }
        )

        e2e_result = benchmark_e2e(
            python_bin=python_bin,
            spec=spec,
            data_path=data_path,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            sample_cap_consistency=args.sample_cap_consistency,
            orig_knn_idx=orig_knn_idx,
        )
        report["groups"][GROUP_E2E_MIXED][spec.name] = e2e_result

        algo_result = benchmark_algo_exact(
            python_bin=python_bin,
            spec=spec,
            data_path=data_path,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            sample_cap_consistency=args.sample_cap_consistency,
            orig_knn_idx=orig_knn_idx,
            idx_path=idx_path,
            dist_path=dist_path,
        )
        report["groups"][GROUP_ALGO_EXACT][spec.name] = algo_result

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(render_markdown(report, python_bin), encoding="utf-8")

    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, indent=2))


if __name__ == "__main__":
    main()
