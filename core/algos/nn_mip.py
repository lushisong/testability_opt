# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from experiments.features import per_test_features

try:
    # 复用 core TinyMLP（纯 numpy），以满足“神经网络方法”的要求
    from core.algos.nn_guided import TinyMLP
except Exception:  # 兜底：若导入失败，定义一个极简线性层
    class TinyMLP:
        def __init__(self, in_dim, hidden=16, lr=1e-2, seed=None):
            rng = np.random.default_rng(seed)
            self.W = rng.normal(0, 0.1, size=(in_dim, 1))
            self.b = np.zeros(1)
            self.lr = lr

        def forward(self, X):
            return X @ self.W + self.b

        def fit_mse(self, X, y, epochs=50, batch=128):
            N = X.shape[0]
            for _ in range(epochs):
                idx = np.arange(N)
                np.random.shuffle(idx)
                for s in range(0, N, batch):
                    part = idx[s:s+batch]
                    pred = self.forward(X[part])
                    err = pred.reshape(-1) - y[part]
                    gW = (X[part].T @ err.reshape(-1, 1)) * (2.0 / max(1, part.size))
                    gb = np.array([2.0 * float(err.mean())])
                    self.W -= self.lr * gW
                    self.b -= self.lr * gb

        def predict(self, X):
            return self.forward(X).reshape(-1)


_MIP_MODEL_CACHE = {}


class NNHintMIPAlgo(BaseAlgo):
    """
    Neural-Diving + CP-SAT (AddHint)
    1) 计算每个测试列的上下文无关特征；
    2) 训练一个轻量 MLP 作为打分器；
    3) 阈值化生成 hint，传入 CP-SAT 求解；
    4) 返回最终可行解与指标。
    """
    name = "NN-MIP"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            time_limit_s: float = 5.0,
            hidden: int = 32, epochs: int = 120, hint_th: float = 0.5,
            budget: float | None = None, num_workers: int = 1, use_callback: bool = False,
            use_cache: bool = True) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        m, n = D.shape

        # 1) 特征（空上下文）与缓存
        ctx = np.zeros(n, dtype=int)
        feats = per_test_features(D, probs, costs, ctx)  # (n, fdim)
        key = (n, _quick_hash(D))
        cached = False
        if use_cache and key in _MIP_MODEL_CACHE:
            net, mu, sd = _MIP_MODEL_CACHE[key]
            cached = True
        else:
            # 2) 简单自监督目标：用 (w_cov + sep_gain)/cost 作为软目标（训练极快）
            num = feats[:, 2] + feats[:, 4]
            den = feats[:, 1] + 1e-6
            y = num / den

            # 归一化 + 轻量 MLP
            mu = feats.mean(axis=0, keepdims=True)
            sd = feats.std(axis=0, keepdims=True) + 1e-8
            Xn = (feats - mu) / sd
            net = TinyMLP(in_dim=Xn.shape[1], hidden=hidden, lr=1e-2, seed=seed)
            net.fit_mse(Xn, y, epochs=epochs, batch=128)
            if use_cache:
                _MIP_MODEL_CACHE[key] = (net, mu, sd)

        Xn = (feats - mu) / sd
        scores = net.predict(Xn)
        prob = 1.0 / (1.0 + np.exp(-scores))
        hint = (prob >= float(hint_th)).astype(int)
        if hint.sum() == 0:
            j = int(np.argmax(y))
            hint[j] = 1

        # 3) 调用 CP-SAT，加上 AddHint
        # 延迟导入 CP-SAT 封装，避免在未安装 ortools 的环境下全局导入失败
        try:
            from experiments.ilp_cp_sat import solve_tp_mip_cp_sat
        except Exception as e:
            raise RuntimeError(
                "NN-MIP 需要安装 OR-Tools (pip install ortools)。\n"
                "请安装依赖或改用 Greedy/Firefly/PSO/NN-Guided。"
            ) from e

        sol = solve_tp_mip_cp_sat(D, probs, costs, tau_d, tau_i,
                                  time_limit_s=time_limit_s, x_hint=hint, log=False,
                                  budget=budget, num_workers=num_workers, use_callback=use_callback)

        selected = sol.get("selected")
        if selected is None:
            selected = np.zeros(n, dtype=int)

        # 4) 打包
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "hint_cnt": int(hint.sum()),
            "cp_status": int(sol.get("status", -1)),
            "feasible": bool(sol.get("feasible", False)),
            "cached": bool(cached),
        })


def _quick_hash(D: np.ndarray) -> int:
    D = (np.asarray(D) > 0).astype(np.uint8)
    return int(hash(D.tobytes()[: min(D.size, 8192)]))
