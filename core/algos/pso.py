# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class BinaryPSOAlgo(BaseAlgo):
    name = "BinaryPSO"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, w=0.7, c1=1.5, c2=1.5,
            penalty=1000.0, w_max: float = 0.9, w_min: float = 0.4,
            v_clamp: float = 4.0, mut_rate: float = 0.01,
            local_refine: bool = True, budget: float | None = None) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n

        X = (rng.random((pop_size, n)) < 0.5).astype(float)
        V = rng.normal(0, 1, size=(pop_size, n))
        P = X.copy()
        Pfit = helper.penalized_objective(P.astype(np.uint8), tau_d, tau_i, penalty)
        g_idx = int(np.argmin(Pfit))
        g = P[g_idx].copy()
        gfit = float(Pfit[g_idx])

        for it in range(max_iter):
            # 惯性权重线性递减
            if w_max is not None and w_min is not None:
                w = float(w_max - (w_max - w_min) * (it / max(1, max_iter - 1)))
            r1 = rng.random((pop_size, n))
            r2 = rng.random((pop_size, n))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
            if v_clamp is not None:
                V = np.clip(V, -abs(float(v_clamp)), abs(float(v_clamp)))
            prob = 1.0 / (1.0 + np.exp(-V))
            X = (rng.random((pop_size, n)) < prob).astype(float)

            fvals = helper.penalized_objective(X.astype(np.uint8), tau_d, tau_i, penalty, budget=budget)
            improved = fvals < Pfit
            if np.any(improved):
                P[improved] = X[improved]
                Pfit[improved] = fvals[improved]

            idx = int(np.argmin(Pfit))
            if Pfit[idx] < gfit:
                gfit = float(Pfit[idx])
                g = P[idx].copy()

            # 变异：对较差个体施加少量随机翻转，避免早熟
            if mut_rate and mut_rate > 0.0:
                worst_k = max(1, pop_size // 10)
                worst_idx = np.argsort(Pfit)[-worst_k:]
                flip_mask = (rng.random((worst_k, n)) < float(mut_rate))
                X[worst_idx] = np.where(flip_mask, 1.0 - X[worst_idx], X[worst_idx])

        # 局部精修：针对全局最优做一次向后清理，去冗余
        gmask = g.astype(np.uint8)
        if local_refine and gmask.sum() > 0:
            tracker = helper.new_tracker()
            tracker.build_from_mask(gmask)
            sel = tracker.selected_mask()
            improved = True
            while improved and sel.any():
                improved = False
                active = np.where(sel == 1)[0]
                for j in active:
                    trial = sel.copy()
                    trial[j] = 0
                    fd_trial, fr_trial, c_trial = helper.evaluate_mask(trial)
                    within_budget = (budget is None or c_trial <= float(budget) + 1e-12)
                    if fd_trial >= tau_d and fr_trial >= tau_i and within_budget:
                        sel = trial
                        improved = True
            gmask = sel

        return BaseAlgo._wrap_result(self.name, gmask.astype(int), D, probs, costs, t0, extra={
            "best_fitness": gfit,
        })
