# -*- coding: utf-8 -*-
import time
import numpy as np
from core.algos.base import BaseAlgo
from core.metrics import penalized_objective

class FireflyAlgo(BaseAlgo):
    name = "Firefly"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, beta0=1.0, gamma=0.6, alpha=0.3,
            penalty=1000.0) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        m, n = D.shape

        def fitness(x):
            return penalized_objective(x, D, probs, costs, tau_d, tau_i, penalty)

        pop = (rng.random((pop_size, n)) < 0.5).astype(int)
        best = None
        best_f = 1e99

        for it in range(max_iter):
            # brightness = -fitness
            fit = np.array([fitness(ind) for ind in pop])
            order = np.argsort(fit)
            if fit[order[0]] < best_f:
                best_f = fit[order[0]]
                best = pop[order[0]].copy()

            # 移动：较暗个体向较亮个体靠拢（按位对齐）
            for i in range(pop_size):
                for j in range(pop_size):
                    if fit[j] < fit[i]:
                        r = np.sum(pop[i] != pop[j]) / max(1, n)  # 归一化Hamming距离
                        beta = beta0 * np.exp(-gamma * (r ** 2))
                        # 向更亮者靠拢：在不同位上以 beta 概率复制更亮个体的位
                        diff_idx = np.where(pop[i] != pop[j])[0]
                        if diff_idx.size > 0:
                            mask = (rng.random(diff_idx.size) < beta)
                            pop[i, diff_idx[mask]] = pop[j, diff_idx[mask]]
                # 随机扰动
                flip_mask = (rng.random(n) < alpha)
                pop[i, flip_mask] = 1 - pop[i, flip_mask]
            # 衰减扰动
            alpha *= 0.98

        from core.algos.base import BaseAlgo
        return BaseAlgo._wrap_result(self.name, best, D, probs, costs, t0, extra={"best_f": best_f})
