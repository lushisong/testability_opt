# -*- coding: utf-8 -*-
import time
import os
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后台绘图，保存图片
import matplotlib.pyplot as plt

from core.metrics import fdr, fir, cost
from core.algos.greedy import GreedyAlgo
from core.algos.firefly import FireflyAlgo
from core.algos.pso import BinaryPSOAlgo
from core.algos.nn_guided import NNGuidedAlgo

ALGO_REGISTRY = {
    "Greedy": GreedyAlgo(),
    "Firefly": FireflyAlgo(),
    "BinaryPSO": BinaryPSOAlgo(),
    "NN-Guided": NNGuidedAlgo(),
}

def run_benchmark(D, probs, costs, tau_d, tau_i, algos: List[str],
                  repeats: int = 10, base_seed: int = 42) -> List[Dict[str, Any]]:
    results = []
    for r in range(repeats):
        for name in algos:
            algo = ALGO_REGISTRY[name]
            t0 = time.perf_counter()
            res = algo.run(D, probs, costs, tau_d, tau_i, seed=base_seed + r)
            results.append({
                "algo": name,
                "repeat": r,
                "fdr": res.fdr,
                "fir": res.fir,
                "cost": res.cost,
                "runtime_sec": res.runtime_sec,
                "selected_cnt": int(res.selected.sum()),
            })
    return results

def summarize_and_plot(results: List[Dict[str, Any]], out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 保存CSV
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    cols = ["algo","repeat","fdr","fir","cost","runtime_sec","selected_cnt"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # 聚合
    algos = sorted({r["algo"] for r in results})
    cost_data = [ [rr["cost"] for rr in results if rr["algo"] == a] for a in algos ]
    time_data = [ [rr["runtime_sec"] for rr in results if rr["algo"] == a] for a in algos ]
    fdr_data  = [ [rr["fdr"] for rr in results if rr["algo"] == a] for a in algos ]
    fir_data  = [ [rr["fir"] for rr in results if rr["algo"] == a] for a in algos ]

    # 图1：成本箱线图
    plt.figure()
    plt.boxplot(cost_data, labels=algos, showmeans=True)
    plt.ylabel("Total Cost")
    plt.title("Cost Distribution by Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_cost.png"))
    plt.close()

    # 图2：运行时间箱线图
    plt.figure()
    plt.boxplot(time_data, labels=algos, showmeans=True)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Distribution by Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_time.png"))
    plt.close()

    # 图3：FDR vs FIR（均值）
    plt.figure()
    means_fdr = [float(np.mean(d)) for d in fdr_data]
    means_fir = [float(np.mean(d)) for d in fir_data]
    x = np.arange(len(algos))
    plt.plot(x, means_fdr, marker="o", label="FDR")
    plt.plot(x, means_fir, marker="s", label="FIR")
    plt.xticks(x, algos)
    plt.ylabel("Score")
    plt.title("Mean FDR/FIR by Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "line_fdr_fir.png"))
    plt.close()

    # 图4：相对最优成本的CDF（近似“性能曲线”）
    all_costs = {a: np.array(c) for a, c in zip(algos, cost_data)}
    best_per_repeat = {}
    # 找同一 repeat 的最优作为归一化基准
    reps = sorted(set(r["repeat"] for r in results))
    rel = {a: [] for a in algos}
    for rep in reps:
        costs_rep = {a: [rr["cost"] for rr in results if rr["algo"] == a and rr["repeat"] == rep] for a in algos}
        base = min(min(v) for v in costs_rep.values())
        for a in algos:
            rel[a].extend([c / base for c in costs_rep[a]])
    plt.figure()
    for a in algos:
        vals = np.sort(np.array(rel[a], dtype=float))
        y = np.linspace(0, 1, len(vals), endpoint=True)
        plt.plot(vals, y, label=a)
    plt.xlabel("Relative Cost to Best (per repeat)")
    plt.ylabel("CDF")
    plt.title("Performance Profile (Relative Cost)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cdf_relative_cost.png"))
    plt.close()

    return {
        "csv": csv_path,
        "figs": [
            "box_cost.png", "box_time.png", "line_fdr_fir.png", "cdf_relative_cost.png"
        ]
    }
