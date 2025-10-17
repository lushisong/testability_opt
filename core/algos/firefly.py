# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class FireflyAlgo(BaseAlgo):
    name = "Firefly"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, beta0=1.0, gamma=0.6, alpha=0.3,
            penalty=1000.0, elite_k: int = 3, budget: float | None = None) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n

        # 初始化种群
        pop = (rng.random((pop_size, n)) < 0.5).astype(np.uint8)
        best = pop[0].copy()
        best_fit = np.inf

        for it in range(max_iter):
            fit = helper.penalized_objective(pop, tau_d, tau_i, penalty, budget=budget)
            order = np.argsort(fit)
            if fit[order[0]] < best_fit:
                best_fit = float(fit[order[0]])
                best = pop[order[0]].copy()

            elites = pop[order[:max(1, min(elite_k, pop_size))]]
            elites_fit = fit[order[:max(1, min(elite_k, pop_size))]]

            # 吸引：非精英个体朝向随机精英移动（基于汉明距离的 beta 衰减）
            for i in range(pop_size):
                j = int(rng.integers(0, elites.shape[0]))
                target = elites[j]
                # 汉明距离归一化
                r_h = float((pop[i] ^ target).mean())
                beta = float(beta0 * np.exp(-gamma * (r_h ** 2)))
                if beta <= 0.0:
                    continue
                move_mask = (rng.random(n) < beta)
                pop[i] = np.where(move_mask, target, pop[i])

            # 随机扰动（退火）
            flip_mask = (rng.random((pop_size, n)) < alpha)
            pop ^= flip_mask.astype(np.uint8)
            alpha *= 0.98  # 退火系数

        # 局部精修：对最优个体做一次向后清理，去除冗余列，并满足预算
        tracker = helper.new_tracker()
        tracker.build_from_mask(best)
        selected = tracker.selected_mask()
        improved = True
        while improved and selected.any():
            improved = False
            active = np.where(selected == 1)[0]
            for j in active:
                trial = selected.copy()
                trial[j] = 0
                fd_trial, fr_trial, c_trial = helper.evaluate_mask(trial)
                within_budget = (budget is None or c_trial <= float(budget) + 1e-12)
                if fd_trial >= tau_d and fr_trial >= tau_i and within_budget:
                    selected = trial
                    improved = True

        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "best_fitness": best_fit,
        })
