# -*- coding: utf-8 -*-
"""
第1步：复现式对比（Firefly vs PSO 等），衡量在 FDR/FIR 阈值下的成本差异。
运行示例：
    python -m experiments.step1_reproduce --m 40 --n 80 --density 0.25 --repeats 20 --tau_d 0.92 --tau_i 0.85
"""
import os, argparse, numpy as np
from core.data_io import random_dataset
from core.benchmark import ALGO_REGISTRY
from experiments.utils import ensure_dir, save_json, performance_profile_relative_cost, tabulate_mean_std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tau_d", type=float, default=0.92)
    ap.add_argument("--tau_i", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="results/step1")
    args = ap.parse_args()

    ensure_dir(args.out)
    rows = []
    algos = ["Firefly", "BinaryPSO", "Greedy", "NN-Guided"]

    for rep in range(args.repeats):
        ds = random_dataset(args.m, args.n, density=args.density, seed=args.seed + rep)
        for name in algos:
            algo = ALGO_REGISTRY[name]
            res = algo.run(ds.D, ds.fault_probs, ds.test_costs, args.tau_d, args.tau_i, seed=args.seed + rep)
            rows.append({
                "algo": name,
                "rep": rep,
                "cost": float(res.cost),
                "fdr": float(res.fdr),
                "fir": float(res.fir),
                "time": float(res.runtime_sec),
                "selected": int(res.selected.sum()),
            })

    # 汇总
    save_json({"rows": rows}, os.path.join(args.out, "rows.json"))

    # 均值±std
    mean_cost = tabulate_mean_std(rows, "cost")
    mean_time = tabulate_mean_std(rows, "time")
    print("=== Cost (mean±std) ===")
    for a, mu, sd in mean_cost:
        print(f"{a:10s}  {mu:.3f} ± {sd:.3f}")
    print("=== Time (s) (mean±std) ===")
    for a, mu, sd in mean_time:
        print(f"{a:10s}  {mu:.3f} ± {sd:.3f}")

    # 计算 Firefly 相对 PSO 的平均改善（成本越低越好）
    ps = [r for r in rows if r["algo"] == "BinaryPSO"]
    ff = [r for r in rows if r["algo"] == "Firefly"]
    if len(ps) == len(ff):
        improv = []
        for i in range(len(ps)):
            base = ps[i]["cost"]
            if base > 0:
                improv.append((base - ff[i]["cost"]) / base)
        if improv:
            print(f"Firefly 相对 PSO 的平均成本降低：{100.0 * float(np.mean(improv)):.1f}%")

    # 性能曲线
    performance_profile_relative_cost(rows, os.path.join(args.out, "perf_profile.png"))
    print(f"[输出] rows.json 与 perf_profile.png 已保存到 {args.out}")

if __name__ == "__main__":
    main()