# -*- coding: utf-8 -*-
import time
import numpy as np
from core.algos.base import BaseAlgo
from core.metrics import penalized_objective

class BinaryPSOAlgo(BaseAlgo):
    name = "BinaryPSO"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, w=0.7, c1=1.5, c2=1.5,
            penalty=1000.0) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        m, n = D.shape

        def fitness(x):
            return penalized_objective(x, D, probs, costs, tau_d, tau_i, penalty)

        # 初始化
        X = (rng.random((pop_size, n)) < 0.5).astype(float)  # 0/1
        V = rng.normal(0, 1, size=(pop_size, n))
        P = X.copy()
        Pfit = np.array([fitness(ind.astype(int)) for ind in P])
        g_idx = np.argmin(Pfit)
        g = P[g_idx].copy()
        gfit = Pfit[g_idx]

        for it in range(max_iter):
            r1 = rng.random((pop_size, n))
            r2 = rng.random((pop_size, n))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
            # 二进制化：sigmoid 概率取1
            prob = 1.0 / (1.0 + np.exp(-V))
            X = (rng.random((pop_size, n)) < prob).astype(float)

            # 评估
            fvals = np.array([fitness(ind.astype(int)) for ind in X])
            # 更新个体最优
            mask = fvals < Pfit
            P[mask] = X[mask]
            Pfit[mask] = fvals[mask]
            # 更新全局最优
            idx = np.argmin(Pfit)
            if Pfit[idx] < gfit:
                gfit = Pfit[idx]
                g = P[idx].copy()

        from core.algos.base import BaseAlgo
        return BaseAlgo._wrap_result(self.name, g.astype(int), D, probs, costs, t0, extra={"best_f": float(gfit)})
