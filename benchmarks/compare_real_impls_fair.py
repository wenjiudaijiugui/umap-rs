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


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"

DATA_DIR = BENCH_DIR / "data_real_fair"
KNN_DIR = BENCH_DIR / "knn_real_fair"
OUT_DIR = BENCH_DIR / "outputs_real_fair"
TIME_DIR = BENCH_DIR / "time_real_fair"

REPORT_JSON = BENCH_DIR / "report_real_fair.json"
REPORT_MD = BENCH_DIR / "report_real_fair.md"
PREVIOUS_REPORT_JSON = BENCH_DIR / "report_real.json"

DEFAULT_PYTHON_BIN = Path(
    os.environ.get(
        "UMAP_BENCH_PYTHON",
        str(Path(sys.executable)),
    )
)
DEFAULT_RSCRIPT_BIN = Path(
    os.environ.get(
        "UMAP_BENCH_RSCRIPT",
        str(shutil.which("Rscript") or "Rscript"),
    )
)

PYTHON_BIN = DEFAULT_PYTHON_BIN
RSCRIPT_BIN = DEFAULT_RSCRIPT_BIN

RUN_PY_E2E = BENCH_DIR / "run_python_umap.py"
RUN_R_E2E = BENCH_DIR / "run_r_uwot.R"
RUN_PY_ALGO = BENCH_DIR / "run_python_umap_algo.py"
RUN_R_ALGO = BENCH_DIR / "run_r_uwot_algo.R"

RUST_FIT = ROOT / "rust_umap" / "target" / "release" / "fit_csv"
RUST_BENCH = ROOT / "rust_umap" / "target" / "release" / "bench_fit_csv"

N_NEIGHBORS = 15
N_COMPONENTS = 2
N_EPOCHS = 200
INIT = "random"

IMPLS = ["python_umap_learn", "r_uwot", "rust_umap"]

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
    p = argparse.ArgumentParser(description="Fair UMAP benchmark with repeats/warmup/random order/shared exact-kNN")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--large-max-samples", type=int, default=15000)
    p.add_argument("--sample-cap-consistency", type=int, default=2000)
    p.add_argument("--python-bin", default=str(DEFAULT_PYTHON_BIN))
    p.add_argument("--rscript-bin", default=str(DEFAULT_RSCRIPT_BIN))
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
        s = line.strip()
        if not s:
            continue
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("No JSON payload found in command output")


def load_real_datasets(seed: int, large_max_samples: int) -> List[DatasetSpec]:
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

    try:
        x_large = fetch_california_housing().data.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            "fetch_california_housing failed. Please ensure network/cache availability for reproducible large-dataset benchmarking"
        ) from exc

    original_n = int(x_large.shape[0])
    if original_n > large_max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(original_n, size=large_max_samples, replace=False)
        x_large = x_large[idx]
    used_n = int(x_large.shape[0])
    x_large = StandardScaler().fit_transform(x_large).astype(np.float32)

    specs.append(
        DatasetSpec(
            name="california_housing",
            x=x_large,
            original_samples=original_n,
            used_samples=used_n,
            n_features=int(x_large.shape[1]),
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


def e2e_cmd(impl: str, data_path: Path, out_path: Path, seed: int) -> List[str]:
    if impl == "python_umap_learn":
        return [
            str(PYTHON_BIN),
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
    if impl == "r_uwot":
        return [
            str(RSCRIPT_BIN),
            str(RUN_R_E2E),
            str(data_path),
            str(out_path),
            str(N_NEIGHBORS),
            str(N_COMPONENTS),
            str(N_EPOCHS),
            str(seed),
            INIT,
        ]
    if impl == "rust_umap":
        return [
            str(RUST_FIT),
            str(data_path),
            str(out_path),
            str(N_NEIGHBORS),
            str(N_COMPONENTS),
            str(N_EPOCHS),
            str(seed),
            INIT,
            "true",
            "30",
            "10",
            "4096",
            "fit",
        ]
    raise ValueError(f"unknown impl: {impl}")


def algo_exact_cmd(
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
            str(PYTHON_BIN),
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
    if impl == "r_uwot":
        return [
            str(RSCRIPT_BIN),
            str(RUN_R_ALGO),
            str(data_path),
            str(out_path),
            str(N_NEIGHBORS),
            str(N_COMPONENTS),
            str(N_EPOCHS),
            str(seed),
            INIT,
            str(warmup),
            str(repeats),
            str(idx_path),
            str(dist_path),
        ]
    if impl == "rust_umap":
        return [
            str(RUST_BENCH),
            str(data_path),
            str(out_path),
            str(N_NEIGHBORS),
            str(N_COMPONENTS),
            str(N_EPOCHS),
            str(seed),
            INIT,
            "false",
            "30",
            "10",
            "4096",
            str(warmup),
            str(repeats),
            str(idx_path),
            str(dist_path),
        ]
    raise ValueError(f"unknown impl: {impl}")


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
        # When evaluating on a sampled subset, rebuild original-space kNN on that
        # subset so both sides use the same local index space [0, sample_cap).
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

    pairwise = {}
    names = sorted(emb_s.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]
            a = emb_s[a_name]
            b = emb_s[b_name]
            _, _, disparity = procrustes(a, b)
            overlap = knn_overlap(a, b, k)
            pairwise[f"{a_name}__vs__{b_name}"] = {
                "procrustes_disparity": float(disparity),
                "knn_overlap_at_15": float(overlap),
            }

    return {
        "sample_size_for_consistency": int(len(sample_idx)),
        "trustworthiness_at_15": trust,
        "original_knn_recall_at_15": recall,
        "pairwise": pairwise,
    }


def build_rust_binaries() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "fit_csv", "--bin", "bench_fit_csv"],
        cwd=ROOT / "rust_umap",
        check=True,
    )


def benchmark_e2e(
    spec: DatasetSpec,
    data_path: Path,
    seed: int,
    warmup: int,
    repeats: int,
    sample_cap_consistency: int,
    orig_knn_idx: np.ndarray,
) -> Dict[str, object]:
    print(f"[e2e][dataset={spec.name}] warmup={warmup}, repeats={repeats}")
    rng = np.random.default_rng(seed + spec.used_samples + 101)

    raw: Dict[str, Dict[str, List[float]]] = {
        impl: {"elapsed_sec": [], "max_rss_mb": []} for impl in IMPLS
    }
    final_embedding_paths: Dict[str, Path] = {}

    total = warmup + repeats
    for rep in range(total):
        order = list(rng.permutation(IMPLS))
        print(f"  [e2e][{spec.name}] rep={rep+1}/{total} order={order}")
        for impl in order:
            out_path = OUT_DIR / f"{spec.name}__{impl}__e2e__rep{rep}.csv"
            time_path = TIME_DIR / f"{spec.name}__{impl}__e2e__rep{rep}.time.txt"
            run = run_timed(e2e_cmd(impl, data_path, out_path, seed), time_path)
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
            "context": {
                "mode": "e2e_default_ann",
                "metric": "euclidean",
                "precomputed_knn": False,
                "use_approximate_knn": impl == "rust_umap",
            },
        }

    embeddings = {
        impl: np.loadtxt(path, delimiter=",", dtype=np.float32)
        for impl, path in final_embedding_paths.items()
    }
    consistency = compute_consistency(
        x=spec.x,
        embeddings=embeddings,
        seed=seed + spec.used_samples,
        orig_knn_idx=orig_knn_idx,
        k=N_NEIGHBORS,
        sample_cap=sample_cap_consistency,
    )

    return {
        "speed_and_memory": summary,
        "consistency": consistency,
    }


def benchmark_algo_exact(
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
    print(f"[algo_exact][dataset={spec.name}] warmup={warmup}, repeats={repeats}")
    rng = np.random.default_rng(seed + spec.used_samples + 202)
    order = list(rng.permutation(IMPLS))
    print(f"  [algo_exact][{spec.name}] invocation_order={order}")

    summary = {}
    final_embedding_paths: Dict[str, Path] = {}

    for impl in order:
        out_path = OUT_DIR / f"{spec.name}__{impl}__algo_exact.csv"
        time_path = TIME_DIR / f"{spec.name}__{impl}__algo_exact.time.txt"
        cmd = algo_exact_cmd(impl, data_path, out_path, seed, warmup, repeats, idx_path, dist_path)
        run = run_timed(cmd, time_path)
        payload = parse_json_line(run.stdout)
        summary[impl] = {
            "fit_mean_sec": float(payload.get("fit_mean_sec", float("nan"))),
            "fit_std_sec": float(payload.get("fit_std_sec", float("nan"))),
            "fit_times_sec": payload.get("fit_times_sec", []),
            "process_elapsed_sec": run.elapsed_sec,
            "process_max_rss_mb": run.max_rss_mb,
            "context": {
                "mode": payload.get("mode", "fit"),
                "metric": payload.get("metric", "euclidean"),
                "knn_metric": payload.get("knn_metric", payload.get("metric", "euclidean")),
                "precomputed_knn": bool(payload.get("precomputed_knn", True)),
                "warmup": payload.get("warmup", warmup),
                "repeats": payload.get("repeats", repeats),
            },
        }
        final_embedding_paths[impl] = out_path

    # restore canonical impl order
    summary = {impl: summary[impl] for impl in IMPLS}

    embeddings = {
        impl: np.loadtxt(path, delimiter=",", dtype=np.float32)
        for impl, path in final_embedding_paths.items()
    }
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


def compare_with_previous(previous_report: Optional[Dict[str, object]], current_e2e: Dict[str, object]) -> Dict[str, object]:
    if previous_report is None:
        return {"available": False}

    out: Dict[str, object] = {"available": True, "by_dataset": {}}
    prev_speed = previous_report.get("speed_and_memory", {})

    for ds_name, metrics in current_e2e.items():
        if ds_name not in prev_speed:
            continue
        prev_ds = prev_speed[ds_name]
        cur_ds = metrics["speed_and_memory"]
        ds_cmp = {}
        for impl in IMPLS:
            if impl not in prev_ds or impl not in cur_ds:
                continue
            old_elapsed = float(prev_ds[impl]["elapsed_sec"])
            old_rss = float(prev_ds[impl]["max_rss_mb"])
            new_elapsed = float(cur_ds[impl]["elapsed_mean_sec"])
            new_rss = float(cur_ds[impl]["max_rss_mean_mb"])
            ds_cmp[impl] = {
                "old_single_elapsed_sec": old_elapsed,
                "new_repeated_elapsed_mean_sec": new_elapsed,
                "elapsed_ratio_new_over_old": (new_elapsed / old_elapsed) if old_elapsed > 0 else float("nan"),
                "old_single_max_rss_mb": old_rss,
                "new_repeated_max_rss_mean_mb": new_rss,
                "rss_ratio_new_over_old": (new_rss / old_rss) if old_rss > 0 else float("nan"),
            }
        out["by_dataset"][ds_name] = ds_cmp
    return out


def main() -> None:
    global PYTHON_BIN, RSCRIPT_BIN
    args = parse_args()
    PYTHON_BIN = Path(args.python_bin)
    RSCRIPT_BIN = Path(args.rscript_bin)
    ensure_dirs()
    build_rust_binaries()

    datasets = load_real_datasets(seed=args.seed, large_max_samples=args.large_max_samples)

    report: Dict[str, object] = {
        "config": {
            "dataset_type": "real",
            "n_neighbors": N_NEIGHBORS,
            "n_components": N_COMPONENTS,
            "n_epochs": N_EPOCHS,
            "init": INIT,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "randomized_order_per_repeat": True,
            "thread_env": THREAD_ENV,
            "groups": [
                "e2e_default_ann",
                "algo_exact_shared_knn",
            ],
            "metric": "euclidean",
            "datasets": [],
        },
        "groups": {
            "e2e_default_ann": {},
            "algo_exact_shared_knn": {},
        },
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
            spec=spec,
            data_path=data_path,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            sample_cap_consistency=args.sample_cap_consistency,
            orig_knn_idx=orig_knn_idx,
        )
        report["groups"]["e2e_default_ann"][spec.name] = e2e_result

        algo_result = benchmark_algo_exact(
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
        report["groups"]["algo_exact_shared_knn"][spec.name] = algo_result

    previous = None
    if PREVIOUS_REPORT_JSON.exists():
        previous = json.loads(PREVIOUS_REPORT_JSON.read_text(encoding="utf-8"))

    report["comparison_to_previous_report_real"] = compare_with_previous(
        previous_report=previous,
        current_e2e=report["groups"]["e2e_default_ann"],
    )

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Fair Real-Dataset UMAP Benchmark Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"- n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS}, n_epochs={N_EPOCHS}, init={INIT}, seed={args.seed}"
    )
    lines.append(f"- warmup={args.warmup}, repeats={args.repeats}, randomized_order_per_repeat=True")
    lines.append(f"- python_bin={PYTHON_BIN}")
    lines.append(f"- rscript_bin={RSCRIPT_BIN}")
    lines.append("- groups: e2e_default_ann, algo_exact_shared_knn")
    lines.append("- thread pinning: " + ", ".join([f"{k}={v}" for k, v in THREAD_ENV.items()]))
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    for ds in report["config"]["datasets"]:
        lines.append(
            f"- {ds['name']}: n_used={ds['n_samples_used']}, n_original={ds['n_samples_original']}, d={ds['n_features']}"
        )
    lines.append("")

    lines.append("## Group A: e2e_default_ann")
    lines.append("")
    for ds_name in report["groups"]["e2e_default_ann"]:
        ds = report["groups"]["e2e_default_ann"][ds_name]
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
        lines.append("")
        lines.append(f"- sample_size_for_consistency: {c['sample_size_for_consistency']}")
        lines.append("- trustworthiness@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['trustworthiness_at_15'][impl]:.6f}")
        lines.append("- original_knn_recall@15:")
        for impl in IMPLS:
            lines.append(f"  - {impl}: {c['original_knn_recall_at_15'][impl]:.6f}")
        lines.append("")

    lines.append("## Group B: algo_exact_shared_knn")
    lines.append("")
    for ds_name in report["groups"]["algo_exact_shared_knn"]:
        ds = report["groups"]["algo_exact_shared_knn"][ds_name]
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
        lines.append("")

    cmp = report["comparison_to_previous_report_real"]
    if cmp.get("available"):
        lines.append("## Comparison vs Previous Single-Run report_real.json")
        lines.append("")
        for ds_name, ds_cmp in cmp.get("by_dataset", {}).items():
            lines.append(f"### Dataset: {ds_name}")
            lines.append("")
            lines.append("| Implementation | old elapsed (s) | new elapsed mean (s) | new/old elapsed | old RSS (MB) | new RSS mean (MB) | new/old RSS |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for impl in IMPLS:
                if impl not in ds_cmp:
                    continue
                x = ds_cmp[impl]
                lines.append(
                    f"| {impl} | {x['old_single_elapsed_sec']:.3f} | {x['new_repeated_elapsed_mean_sec']:.3f} | {x['elapsed_ratio_new_over_old']:.3f} | {x['old_single_max_rss_mb']:.1f} | {x['new_repeated_max_rss_mean_mb']:.1f} | {x['rss_ratio_new_over_old']:.3f} |"
                )
            lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to:\n  - {REPORT_JSON}\n  - {REPORT_MD}")


if __name__ == "__main__":
    main()
