# -*- coding: utf-8 -*-
import time
import os
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后台绘图，保存图片
import matplotlib.pyplot as plt

from core.metrics import fdr, fir, cost
from core.algos.base import BaseAlgo
from core.algos.greedy import GreedyAlgo
from core.algos.firefly import FireflyAlgo
from core.algos.pso import BinaryPSOAlgo
from core.algos.nn_guided import NNGuidedAlgo
from core.algos.nn_guided_offline import NNGuidedOfflineAlgo
from core.algos.nn_mip import NNHintMIPAlgo
from core.algos.nn_mip_offline import NNHintMIPOfflineAlgo

AlgoFactory = Callable[[], BaseAlgo]

ALGO_REGISTRY: Dict[str, AlgoFactory | type[BaseAlgo]] = {
    "Greedy": GreedyAlgo,
    "Firefly": FireflyAlgo,
    "BinaryPSO": BinaryPSOAlgo,
    "NN-Guided": NNGuidedAlgo,
    "NN-MIP": NNHintMIPAlgo,
    "NN-Guided_Offline": NNGuidedOfflineAlgo,
    "NN-MIP_Offline": NNHintMIPOfflineAlgo,
}


def _instantiate_algo(name: str) -> BaseAlgo:
    factory = ALGO_REGISTRY[name]
    if isinstance(factory, BaseAlgo):
        # defensive copy via class constructor if available
        return factory.__class__()
    if isinstance(factory, type):
        return factory()
    algo = factory()
    if not isinstance(algo, BaseAlgo):
        raise TypeError(f"Factory for {name!r} did not yield BaseAlgo instance")
    return algo


def create_algo(name: str) -> BaseAlgo:
    """Public helper to instantiate algorithms registered in ``ALGO_REGISTRY``."""

    return _instantiate_algo(name)


def _normalize_trajectory(result) -> List[Dict[str, float]] | None:
    traj = result.trajectory
    if traj is None:
        return None
    normalized = []
    for entry in traj:
        if isinstance(entry, dict):
            normalized.append({
                "time": float(entry.get("time", entry.get("t", 0.0))),
                "objective": float(entry.get("objective", entry.get("cost", 0.0))),
                "fdr": float(entry.get("fdr", entry.get("fdr_estimate", 0.0))),
                "fir": float(entry.get("fir", entry.get("fir_estimate", 0.0))),
            })
        elif isinstance(entry, (tuple, list)) and entry:
            time_val = float(entry[0])
            obj_val = float(entry[1]) if len(entry) > 1 else float("nan")
            normalized.append({
                "time": time_val,
                "objective": obj_val,
                "fdr": float(entry[2]) if len(entry) > 2 else float("nan"),
                "fir": float(entry[3]) if len(entry) > 3 else float("nan"),
            })
    normalized.sort(key=lambda x: x["time"])
    return normalized if normalized else None


def _run_single_benchmark(
    job: Tuple[str, int, int, Any, Any, Any, float, float, float | None]
) -> Dict[str, Any]:
    name, repeat, base_seed, D, probs, costs, tau_d, tau_i, budget = job
    algo = _instantiate_algo(name)
    seed = base_seed + repeat
    t0 = time.perf_counter()
    try:
        res = algo.run(D, probs, costs, tau_d, tau_i, seed=seed, budget=budget)
    except TypeError:
        res = algo.run(D, probs, costs, tau_d, tau_i, seed=seed)
    runtime_overhead = time.perf_counter() - t0
    cache_hits = res.cache_hits
    cache_queries = res.cache_queries
    if cache_hits is None:
        cache_hits = res.extra.get("cache_hits") or res.extra.get("model_cache_hits")
    if cache_queries is None:
        cache_queries = res.extra.get("cache_queries") or res.extra.get("model_cache_queries")
    cache_rate = None
    if cache_hits is not None:
        cache_hits = int(cache_hits)
    if cache_queries is not None:
        cache_queries = int(cache_queries)
    if cache_hits is not None and cache_queries:
        cache_rate = float(cache_hits) / float(cache_queries)
    trajectory = _normalize_trajectory(res)
    if trajectory is None:
        trajectory = [{
            "time": float(res.runtime_sec),
            "objective": float(res.cost),
            "fdr": float(res.fdr),
            "fir": float(res.fir),
        }]
    primal_gap = res.primal_gap if res.primal_gap is not None else max(0.0, tau_d - float(res.fdr))
    dual_gap = res.dual_gap if res.dual_gap is not None else max(0.0, tau_i - float(res.fir))
    entry = {
        "algo": name,
        "repeat": repeat,
        "seed": seed,
        "fdr": float(res.fdr),
        "fir": float(res.fir),
        "cost": float(res.cost),
        "runtime_sec": float(res.runtime_sec),
        "selected_cnt": int(res.selected.sum()),
        "budget": float(budget) if budget is not None else None,
        "primal_gap": float(primal_gap),
        "dual_gap": float(dual_gap),
        "cache_hits": cache_hits,
        "cache_queries": cache_queries,
        "cache_hit_rate": cache_rate,
        "trajectory": trajectory,
        "overhead_sec": runtime_overhead,
    }
    if res.extra:
        entry["extra"] = res.extra
    return entry

def run_benchmark(
    D,
    probs,
    costs,
    tau_d,
    tau_i,
    algos: List[str],
    repeats: int = 10,
    base_seed: int = 42,
    budget: float | None = None,
    scheduler: str = "sequential",
    max_workers: int | None = None,
) -> List[Dict[str, Any]]:
    jobs = [
        (name, repeat, base_seed, D, probs, costs, tau_d, tau_i, budget)
        for repeat in range(repeats)
        for name in algos
    ]
    if not jobs:
        return []

    results: List[Dict[str, Any]] = []
    scheduler = (scheduler or "sequential").lower()
    if scheduler == "sequential" or len(jobs) == 1:
        for job in jobs:
            results.append(_run_single_benchmark(job))
    else:
        executor_cls = {
            "thread": ThreadPoolExecutor,
            "process": ProcessPoolExecutor,
        }.get(scheduler)
        if executor_cls is None:
            raise ValueError(f"Unknown scheduler '{scheduler}'. Expected 'sequential', 'thread', or 'process'.")
        with executor_cls(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_benchmark, job): job for job in jobs}
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: (r["repeat"], r["algo"]))
    return results

def summarize_and_plot(results: List[Dict[str, Any]], out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not results:
        return {"csv": None, "figs": []}

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    cols = [
        "algo",
        "repeat",
        "seed",
        "budget",
        "fdr",
        "fir",
        "cost",
        "runtime_sec",
        "selected_cnt",
        "primal_gap",
        "dual_gap",
        "cache_hit_rate",
        "cache_hits",
        "cache_queries",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in results:
            f.write(",".join(str(row.get(col, "")) for col in cols) + "\n")

    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    algos = sorted({r["algo"] for r in results})
    cost_data = [[rr["cost"] for rr in results if rr["algo"] == a] for a in algos]
    time_data = [[rr["runtime_sec"] for rr in results if rr["algo"] == a] for a in algos]
    fdr_data = [[rr["fdr"] for rr in results if rr["algo"] == a] for a in algos]
    fir_data = [[rr["fir"] for rr in results if rr["algo"] == a] for a in algos]
    primal_gap_data = [[rr["primal_gap"] for rr in results if rr["algo"] == a] for a in algos]
    dual_gap_data = [[rr["dual_gap"] for rr in results if rr["algo"] == a] for a in algos]

    plt.figure()
    plt.boxplot(cost_data, labels=algos, showmeans=True)
    plt.ylabel("Total Cost")
    plt.title("Cost Distribution by Algorithm")
    plt.tight_layout()
    box_cost = os.path.join(out_dir, "box_cost.png")
    plt.savefig(box_cost)
    plt.close()

    plt.figure()
    plt.boxplot(time_data, labels=algos, showmeans=True)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Distribution by Algorithm")
    plt.tight_layout()
    box_time = os.path.join(out_dir, "box_time.png")
    plt.savefig(box_time)
    plt.close()

    plt.figure()
    means_fdr = [float(np.mean(d)) if d else 0.0 for d in fdr_data]
    means_fir = [float(np.mean(d)) if d else 0.0 for d in fir_data]
    x = np.arange(len(algos))
    plt.plot(x, means_fdr, marker="o", label="FDR")
    plt.plot(x, means_fir, marker="s", label="FIR")
    plt.xticks(x, algos)
    plt.ylabel("Score")
    plt.title("Mean FDR/FIR by Algorithm")
    plt.legend()
    plt.tight_layout()
    line_fdr_fir = os.path.join(out_dir, "line_fdr_fir.png")
    plt.savefig(line_fdr_fir)
    plt.close()

    reps = sorted(set(r["repeat"] for r in results))
    rel = {a: [] for a in algos}
    for rep in reps:
        sub = [r for r in results if r["repeat"] == rep]
        best = min(r["cost"] for r in sub)
        for a in algos:
            rel[a].extend([r["cost"] / (best + 1e-12) for r in sub if r["algo"] == a])
    plt.figure()
    for a in algos:
        vals = np.sort(np.array(rel[a], dtype=float))
        if vals.size == 0:
            continue
        y = np.linspace(0, 1, len(vals), endpoint=True)
        plt.plot(vals, y, label=a)
    plt.xlabel("Relative Cost to Best (per repeat)")
    plt.ylabel("CDF")
    plt.title("Performance Profile (Relative Cost)")
    plt.legend()
    plt.tight_layout()
    perf_profile = os.path.join(out_dir, "performance_profile_cost.png")
    plt.savefig(perf_profile)
    plt.close()

    plt.figure()
    for a in algos:
        vals = np.sort(np.array([r["runtime_sec"] for r in results if r["algo"] == a], dtype=float))
        if vals.size == 0:
            continue
        y = np.linspace(0, 1, len(vals), endpoint=True)
        plt.plot(vals, y, label=a)
    plt.xlabel("Runtime (s)")
    plt.ylabel("CDF")
    plt.title("Runtime Cumulative Distribution")
    plt.legend()
    plt.tight_layout()
    runtime_cdf = os.path.join(out_dir, "cdf_runtime.png")
    plt.savefig(runtime_cdf)
    plt.close()

    def _mean_ci(data: List[float]) -> Tuple[float, float]:
        if not data:
            return 0.0, 0.0
        mean = float(np.mean(data))
        if len(data) <= 1:
            return mean, 0.0
        std = float(np.std(data, ddof=1))
        margin = 1.96 * std / math.sqrt(len(data))
        return mean, margin

    fdr_stats = [_mean_ci(d) for d in fdr_data]
    fir_stats = [_mean_ci(d) for d in fir_data]
    plt.figure()
    width = 0.35
    idx = np.arange(len(algos))
    plt.bar(idx - width / 2, [m for m, _ in fdr_stats], width, yerr=[c for _, c in fdr_stats], label="FDR", capsize=4)
    plt.bar(idx + width / 2, [m for m, _ in fir_stats], width, yerr=[c for _, c in fir_stats], label="FIR", capsize=4)
    plt.xticks(idx, algos)
    plt.ylabel("Score")
    plt.title("FDR/FIR with 95% CI")
    plt.legend()
    plt.tight_layout()
    ci_fig = os.path.join(out_dir, "ci_fdr_fir.png")
    plt.savefig(ci_fig)
    plt.close()

    traj_per_algo: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in results:
        for pt in row.get("trajectory", []) or []:
            t = float(pt.get("time", 0.0))
            obj = float(pt.get("objective", pt.get("cost", row.get("cost", 0.0))))
            traj_per_algo[row["algo"]].append((t, obj))
    anytime = {}
    for algo, points in traj_per_algo.items():
        if not points:
            anytime[algo] = []
            continue
        points.sort(key=lambda x: x[0])
        best = math.inf
        series = []
        for t, obj in points:
            if math.isfinite(obj):
                best = min(best, obj)
            if not math.isfinite(best):
                best = obj
            series.append((t, best))
        anytime[algo] = series

    anytime_path = os.path.join(out_dir, "anytime_trajectories.json")
    with open(anytime_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in anytime.items()}, f, ensure_ascii=False, indent=2)

    plt.figure()
    for algo, series in anytime.items():
        if not series:
            continue
        times = [p[0] for p in series]
        values = [p[1] for p in series]
        plt.step(times, values, where="post", label=algo)
    plt.xlabel("Time (s)")
    plt.ylabel("Best Cost so far")
    plt.title("Anytime Cost Trajectory")
    plt.legend()
    plt.tight_layout()
    anytime_fig = os.path.join(out_dir, "anytime_profile.png")
    plt.savefig(anytime_fig)
    plt.close()

    def _algo_category(name: str) -> Dict[str, bool]:
        lower = name.lower()
        return {
            "offline": "offline" in lower,
            "neural": "nn" in lower,
        }

    categories = {
        "offline": lambda n: _algo_category(n)["offline"],
        "online": lambda n: not _algo_category(n)["offline"],
        "neural": lambda n: _algo_category(n)["neural"],
        "classical": lambda n: not _algo_category(n)["neural"],
    }
    metrics = ["cost", "runtime_sec", "fdr", "fir", "primal_gap", "dual_gap"]
    category_summary = {}
    for label, predicate in categories.items():
        subset = [row for row in results if predicate(row["algo"])]
        if not subset:
            continue
        category_summary[label] = {
            metric: float(np.mean([row[metric] for row in subset])) for metric in metrics
        }
        category_summary[label]["samples"] = len(subset)

    overall = {}
    for algo in algos:
        subset = [row for row in results if row["algo"] == algo]
        stats = {}
        for metric in metrics:
            values = [row[metric] for row in subset]
            mean, margin = _mean_ci(values)
            stats[metric] = {
                "mean": mean,
                "ci95": margin,
            }
        stats["count"] = len(subset)
        overall[algo] = stats

    report = {
        "overall": overall,
        "categories": category_summary,
    }
    report_path = os.path.join(out_dir, "summary_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fig_paths = [
        box_cost,
        box_time,
        line_fdr_fir,
        perf_profile,
        runtime_cdf,
        ci_fig,
        anytime_fig,
    ]

    return {
        "csv": csv_path,
        "json": json_path,
        "anytime": anytime_path,
        "report": report_path,
        "figs": [os.path.basename(p) for p in fig_paths],
    }
