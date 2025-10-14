# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class GreedyAlgo(BaseAlgo):
    name = "Greedy"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            w_fdr=0.5, w_fir=0.5, backward_cleanup=True) -> "AlgoResult":
        t0 = time.perf_counter()
        helper = BinaryMetricHelper(D, probs, costs)
        tracker = helper.new_tracker()
        rng = np.random.default_rng(seed)

        while True:
            fd = tracker.current_fdr()
            fr = tracker.current_fir()
            if fd >= tau_d and fr >= tau_i:
                break

            fdr_gain, fir_gain = tracker.gain_vectors()
            score = w_fdr * fdr_gain + w_fir * fir_gain
            denom = np.maximum(helper.costs, 1e-12)
            score = score / denom
            score[tracker.selected == 1] = -np.inf

            best_j = int(np.argmax(score))
            if not np.isfinite(score[best_j]) or score[best_j] <= 0.0:
                candidates = np.where(tracker.selected == 0)[0]
                if candidates.size == 0:
                    break
                # tie-break by minimum cost then random choice to avoid bias
                min_cost = helper.costs[candidates].min()
                cheapest = candidates[np.isclose(helper.costs[candidates], min_cost)]
                best_j = int(rng.choice(cheapest))

            tracker.add(best_j)

        selected = tracker.selected_mask()

        if backward_cleanup and selected.any():
            improved = True
            while improved:
                improved = False
                active = np.where(selected == 1)[0]
                for j in active:
                    trial = selected.copy()
                    trial[j] = 0
                    fd_trial, fr_trial, _ = helper.evaluate_mask(trial)
                    if fd_trial >= tau_d and fr_trial >= tau_i:
                        selected = trial
                        tracker.build_from_mask(selected)
                        improved = True

        result = helper.evaluate_mask(selected)
        # Ensure computed metrics match tracker state (wrap_result recomputes for safety)
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "fdr_estimate": result[0],
            "fir_estimate": result[1],
        })
