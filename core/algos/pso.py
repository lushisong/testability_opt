# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class BinaryPSOAlgo(BaseAlgo):
    name = "BinaryPSO"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            pop_size=30, max_iter=200, w=0.7, c1=1.5, c2=1.5,
            penalty=1000.0) -> "AlgoResult":
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

        for _ in range(max_iter):
            r1 = rng.random((pop_size, n))
            r2 = rng.random((pop_size, n))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
            prob = 1.0 / (1.0 + np.exp(-V))
            X = (rng.random((pop_size, n)) < prob).astype(float)

            fvals = helper.penalized_objective(X.astype(np.uint8), tau_d, tau_i, penalty)
            improved = fvals < Pfit
            if np.any(improved):
                P[improved] = X[improved]
                Pfit[improved] = fvals[improved]

            idx = int(np.argmin(Pfit))
            if Pfit[idx] < gfit:
                gfit = float(Pfit[idx])
                g = P[idx].copy()

        return BaseAlgo._wrap_result(self.name, g.astype(int), D, probs, costs, t0, extra={
            "best_fitness": gfit,
        })
