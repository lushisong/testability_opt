# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class FireflyAlgo(BaseAlgo):
    name = "Firefly"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, beta0=1.0, gamma=0.6, alpha=0.3,
            penalty=1000.0) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n

        pop = (rng.random((pop_size, n)) < 0.5).astype(np.uint8)
        best = pop[0].copy()
        best_fit = np.inf

        for _ in range(max_iter):
            fit = helper.penalized_objective(pop, tau_d, tau_i, penalty)
            idx_best = int(np.argmin(fit))
            if fit[idx_best] < best_fit:
                best_fit = float(fit[idx_best])
                best = pop[idx_best].copy()

            diff = pop ^ best
            r = diff.mean(axis=1)
            beta = beta0 * np.exp(-gamma * (r ** 2))
            beta[idx_best] = 0.0
            move_mask = rng.random((pop_size, n)) < beta[:, None]
            pop = np.where(move_mask, best, pop)

            flip_mask = rng.random((pop_size, n)) < alpha
            pop ^= flip_mask.astype(np.uint8)
            alpha *= 0.98

        return BaseAlgo._wrap_result(self.name, best, D, probs, costs, t0, extra={
            "best_fitness": best_fit,
        })
