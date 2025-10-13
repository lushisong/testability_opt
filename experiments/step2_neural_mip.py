# -*- coding: utf-8 -*-
"""
第2步：Neural-Diving + MILP（CP-SAT）用于测点优化
流程：
1) 用 CP-SAT（无 Hint）在若干训练实例上求解，收集 teacher 的 x*；
2) 将每个候选测试列的上下文无关特征（或空上下文）与 teacher 的选取标签配对，训练 TinyMLP；
3) 对测试实例用 NN 预测 x 的初值/提示（diving hint），传入 CP-SAT 的 AddHint；
4) 对比“无 Hint vs NN Hint”在相同时间上限下的成本、时间、anytime 轨迹等指标；
5) 输出 JSON 与图片，以便后续论文撰写与参数调优。

注意：
- anytime 轨迹来自 CP-SAT 的解回调，记录的是“时间-目标值”的演化。由于回调层无法直接取连续代价，
  本实现将回调中的 objective value 近似视作成本的单调代理，图标题标注为“approx.”。

运行示例：

python -m experiments.step2_neural_mip --train_n 60 --train_m 30 --train_k 120 \
    --test_n 60 --test_m 30 --test_k 40 --tau_d 0.92 --tau_i 0.85 --time_limit 10
    
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.data_io import random_dataset
from experiments.features import per_test_features
from experiments.models import TinyMLP
from experiments.ilp_cp_sat import solve_tp_mip_cp_sat
from experiments.utils import ensure_dir, anytime_plot

def collect_teacher_pairs(num_inst: int, m: int, n: int, density: float,
                          tau_d: float, tau_i: float, time_limit: float, seed: int):
    """
    用 CP-SAT（无 Hint）在 num_inst 个随机实例上生成 teacher 选点 x*，
    将“空上下文特征”与 x* 的每一列构成监督样本。
    返回：X (num_inst*n, fdim), y (num_inst*n,)
    """
    X_list, y_list = [], []
    for k in range(num_inst):
        ds = random_dataset(m, n, density=density, seed=seed + k)
        sol = solve_tp_mip_cp_sat(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i,
                                  time_limit_s=time_limit, x_hint=None, log=False)
        x_star = sol["selected"]  # (n,)
        # 采用空上下文（selected=0向量）计算列特征，学习“静态重要度”
        ctx = np.zeros(n, dtype=int)
        feats = per_test_features(ds.D, ds.fault_probs, ds.test_costs, ctx)  # (n, fdim)
        X_list.append(feats)
        y_list.append(x_star.reshape(-1, 1))
    X = np.vstack(X_list).astype(float)               # (num_inst*n, fdim)
    y = np.vstack(y_list).reshape(-1).astype(float)   # (num_inst*n,)
    return X, y

def train_nn(X, y, epochs=300, hidden=32, seed=0):
    """
    训练一个轻量两层 MLP，对每列测试给出“被 teacher 选择”的概率打分。
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sd
    net = TinyMLP(in_dim=X.shape[1], hidden=hidden, lr=1e-2, seed=seed)
    net.fit_mse(Xn, y, epochs=epochs, batch=256)
    return net, mu, sd

def predict_hint(net, mu, sd, D, probs, costs, threshold=0.5):
    """
    用训练好的 MLP 对新实例的每个测试列进行打分，并阈值化形成 {0,1} Hint。
    为避免全 0，若没有任何列超过阈值，则选择一个性价比最高的列。
    """
    n = D.shape[1]
    ctx = np.zeros(n, dtype=int)  # 空上下文
    feats = per_test_features(D, probs, costs, ctx)         # (n, fdim)
    scores = net.predict((feats - mu) / sd)                 # 实值
    prob = 1.0 / (1.0 + np.exp(-scores))                    # Sigmoid 概率
    hint = (prob >= threshold).astype(int)
    if hint.sum() == 0:
        # 退化处理：按“覆盖权重+区分增益”/成本的比值选一个
        num = feats[:, 2] + feats[:, 4]     # w_cov + sep_gain
        den = feats[:, 1] + 1e-6            # cost
        j = int(np.argmax(num / den))
        hint[j] = 1
    return hint, prob

def boxplot_pair(data_a, data_b, labels, title, ylabel, out_png):
    plt.figure()
    plt.boxplot([data_a, data_b], labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    # 训练集
    ap.add_argument("--train_k", type=int, default=120, help="训练实例数量")
    ap.add_argument("--train_m", type=int, default=30)
    ap.add_argument("--train_n", type=int, default=60)
    # 测试集
    ap.add_argument("--test_k", type=int, default=40, help="测试实例数量")
    ap.add_argument("--test_m", type=int, default=30)
    ap.add_argument("--test_n", type=int, default=60)
    # 通用
    ap.add_argument("--density", type=float, default=0.30)
    ap.add_argument("--tau_d", type=float, default=0.92)
    ap.add_argument("--tau_i", type=float, default=0.85)
    ap.add_argument("--time_limit", type=float, default=10.0, help="CP-SAT 单实例求解时间上限(秒)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="results/step2")
    # NN 超参
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--hint_th", type=float, default=0.5)
    args = ap.parse_args()

    ensure_dir(args.out)

    # 1) Teacher 采样并训练 NN
    print("[Step2] 收集 teacher 样本并训练 NN ...")
    X, y = collect_teacher_pairs(args.train_k, args.train_m, args.train_n, args.density,
                                 args.tau_d, args.tau_i, args.time_limit, args.seed)
    net, mu, sd = train_nn(X, y, epochs=args.epochs, hidden=args.hidden, seed=args.seed)
    print(f"[Step2] 训练完成。训练样本数={len(y)}, 正样本比例={float((y>0.5).mean()):.3f}")

    # 2) 测试：对每个实例比较 无Hint vs NN Hint
    costs_no, costs_nn = [], []
    times_no, times_nn = [], []
    feas_no, feas_nn = [], []
    # 任取第一个实例作 anytime 轨迹示例
    traj = {"CP-SAT (no hint)": None, "CP-SAT (NN hint)": None}

    for k in range(args.test_k):
        ds = random_dataset(args.test_m, args.test_n, density=args.density, seed=args.seed + 10_000 + k)

        # 无 Hint
        sol0 = solve_tp_mip_cp_sat(ds.D, ds.fault_probs, ds.test_costs, args.tau_d, args.tau_i,
                                   time_limit_s=args.time_limit, x_hint=None, log=False)
        costs_no.append(float(sol0["objective_cost"]))
        t0 = sol0["anytime_traj"][-1][0] if sol0["anytime_traj"] else float(args.time_limit)
        times_no.append(t0)
        feas_no.append(bool(sol0["feasible"]))
        if k == 0:
            traj["CP-SAT (no hint)"] = sol0["anytime_traj"]

        # NN Hint
        hint, prob = predict_hint(net, mu, sd, ds.D, ds.fault_probs, ds.test_costs, threshold=args.hint_th)
        sol1 = solve_tp_mip_cp_sat(ds.D, ds.fault_probs, ds.test_costs, args.tau_d, args.tau_i,
                                   time_limit_s=args.time_limit, x_hint=hint, log=False)
        costs_nn.append(float(sol1["objective_cost"]))
        t1 = sol1["anytime_traj"][-1][0] if sol1["anytime_traj"] else float(args.time_limit)
        times_nn.append(t1)
        feas_nn.append(bool(sol1["feasible"]))
        if k == 0:
            traj["CP-SAT (NN hint)"] = sol1["anytime_traj"]

    costs_no = np.array(costs_no, dtype=float)
    costs_nn = np.array(costs_nn, dtype=float)
    times_no = np.array(times_no, dtype=float)
    times_nn = np.array(times_nn, dtype=float)
    feas_no = np.array(feas_no, dtype=bool)
    feas_nn = np.array(feas_nn, dtype=bool)

    # 3) 统计与输出
    out_json = {
        "args": vars(args),
        "summary": {
            "cost_no_mean": float(costs_no.mean()),
            "cost_nn_mean": float(costs_nn.mean()),
            "time_no_mean": float(times_no.mean()),
            "time_nn_mean": float(times_nn.mean()),
            "feasible_no_rate": float(feas_no.mean()),
            "feasible_nn_rate": float(feas_nn.mean()),
            "cost_improve_mean_pct": float(100.0 * (costs_no.mean() - costs_nn.mean()) / (costs_no.mean() + 1e-12)),
            "time_improve_mean_pct": float(100.0 * (times_no.mean() - times_nn.mean()) / (times_no.mean() + 1e-12)),
        }
    }
    # 单实例级别的明细
    out_json["per_instance"] = [
        {
            "k": int(k),
            "cost_no": float(costs_no[k]),
            "cost_nn": float(costs_nn[k]),
            "time_no": float(times_no[k]),
            "time_nn": float(times_nn[k]),
            "feasible_no": bool(feas_no[k]),
            "feasible_nn": bool(feas_nn[k]),
        }
        for k in range(args.test_k)
    ]

    import json
    with open(os.path.join(args.out, "step2_results.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print("=== Step2 汇总 ===")
    print(f"无 Hint:  mean cost={costs_no.mean():.3f}, mean time={times_no.mean():.3f}s, feasible rate={feas_no.mean():.2f}")
    print(f"NN Hint: mean cost={costs_nn.mean():.3f}, mean time={times_nn.mean():.3f}s, feasible rate={feas_nn.mean():.2f}")
    print(f"平均成本改善: {100.0 * (costs_no.mean() - costs_nn.mean()) / (costs_no.mean() + 1e-12):.1f}%")
    print(f"平均时间改善: {100.0 * (times_no.mean() - times_nn.mean()) / (times_no.mean() + 1e-12):.1f}%")

    # 4) 作图：anytime、箱线图、散点与改进条形
    # 4.1 Anytime 轨迹（注意：纵轴近似值，仅用于趋势展示）
    anytime_plot(traj, os.path.join(args.out, "anytime_approx.png"),
                 xlabel="Time (s)", ylabel="Objective (approx.)")

    # 4.2 成本与时间箱线图
    boxplot_pair(costs_no, costs_nn, labels=["CP-SAT (no hint)", "CP-SAT (NN hint)"],
                 title="Cost Distribution (lower is better)",
                 ylabel="Total Cost",
                 out_png=os.path.join(args.out, "box_cost_step2.png"))
    boxplot_pair(times_no, times_nn, labels=["CP-SAT (no hint)", "CP-SAT (NN hint)"],
                 title="Runtime Distribution",
                 ylabel="Time (s)",
                 out_png=os.path.join(args.out, "box_time_step2.png"))

    # 4.3 每实例相对改进直方图（成本）
    rel_improve = (costs_no - costs_nn) / (costs_no + 1e-12)
    plt.figure()
    plt.hist(rel_improve, bins=20, edgecolor="black")
    plt.xlabel("Relative Cost Improvement (positive = better)")
    plt.ylabel("Count")
    plt.title("Histogram of Relative Cost Improvement")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "hist_rel_cost_improve.png"))
    plt.close()

    print(f"[输出] JSON 与图片已保存至：{args.out}")

if __name__ == "__main__":
    main()
