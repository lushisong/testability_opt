# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


class GreedyAlgo(BaseAlgo):
    name = "Greedy"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            w_fdr=0.5, w_fir=0.5, backward_cleanup=True,
            dynamic_weight=True, cost_power: float = 1.0,
            budget: float | None = None) -> "AlgoResult":
        t0 = time.perf_counter()
        helper = BinaryMetricHelper(D, probs, costs)
        tracker = helper.new_tracker()
        rng = np.random.default_rng(seed)

        blocked = np.zeros(helper.n, dtype=bool)
        while True:
            fd = tracker.current_fdr()
            fr = tracker.current_fir()
            if fd >= tau_d and fr >= tau_i:
                break

            fdr_gain, fir_gain = tracker.gain_vectors()
            # 动态权重：聚焦当前短板（gap）
            if dynamic_weight:
                gd = max(0.0, tau_d - fd)
                gi = max(0.0, tau_i - fr)
                s = gd + gi + 1e-12
                wf = gd / s
                wr = gi / s
                score = wf * fdr_gain + wr * fir_gain
            else:
                score = w_fdr * fdr_gain + w_fir * fir_gain
            # 成本缩放：score / cost^alpha
            denom = np.maximum(helper.costs ** max(0.0, float(cost_power)), 1e-12)
            score = score / denom
            score[(tracker.selected == 1) | blocked] = -np.inf

            best_j = int(np.argmax(score))
            if not np.isfinite(score[best_j]) or score[best_j] <= 0.0:
                candidates = np.where((tracker.selected == 0) & (~blocked))[0]
                if candidates.size == 0:
                    break
                # tie-break by minimum cost then random choice to avoid bias
                min_cost = helper.costs[candidates].min()
                cheapest = candidates[np.isclose(helper.costs[candidates], min_cost)]
                best_j = int(rng.choice(cheapest))

            # 预算约束：添加前检查
            if budget is not None:
                cur_cost = float(tracker.selected.astype(float) @ helper.costs)
                if cur_cost + float(helper.costs[best_j]) > float(budget) + 1e-12:
                    # 标记为已阻止，避免死循环，但不加入选择集
                    blocked[best_j] = True
                    continue
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
                    fd_trial, fr_trial, c_trial = helper.evaluate_mask(trial)
                    within_budget = (budget is None or c_trial <= float(budget) + 1e-12)
                    if fd_trial >= tau_d and fr_trial >= tau_i and within_budget:
                        selected = trial
                        tracker.build_from_mask(selected)
                        improved = True

        result = helper.evaluate_mask(selected)
        # Ensure computed metrics match tracker state (wrap_result recomputes for safety)
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "fdr_estimate": result[0],
            "fir_estimate": result[1],
        })
