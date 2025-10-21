# -*- coding: utf-8 -*-
import os
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper, structure_distance, structure_profile
from experiments.models import load_tinymlp


class NNGuidedOfflineAlgo(BaseAlgo):
    name = "NN-Guided_Offline"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            model_path: str = os.path.join("data", "models", "nn_guided_offline.npz"),
            budget: float | None = None, structure_tolerance: float = 0.1) -> "AlgoResult":
        t0 = time.perf_counter()
        helper = BinaryMetricHelper(D, probs, costs)
        profile = structure_profile(D)
        profile_distance = float("inf")

        if not os.path.exists(model_path):
            # 回退到在线版本
            from core.algos.nn_guided import NNGuidedAlgo
            res = NNGuidedAlgo().run(D, probs, costs, tau_d, tau_i, seed=seed, budget=budget)
            return BaseAlgo._wrap_result(
                self.name,
                res.selected,
                D,
                probs,
                costs,
                t0,
                extra={
                    "fallback": True,
                    "profile_distance": profile_distance,
                    "structure_tolerance": float(structure_tolerance),
                    "model_path": model_path,
                },
            )

        net, mu, sd, meta = load_tinymlp(model_path)
        saved_profile = meta.get("profile")
        if saved_profile is not None:
            profile_distance = structure_distance(saved_profile, profile)
        if saved_profile is None or profile_distance > structure_tolerance:
            from core.algos.nn_guided import NNGuidedAlgo
            res = NNGuidedAlgo().run(
                D,
                probs,
                costs,
                tau_d,
                tau_i,
                seed=seed,
                budget=budget,
                use_cache=True,
                structure_tolerance=structure_tolerance,
            )
            return BaseAlgo._wrap_result(
                self.name,
                res.selected,
                D,
                probs,
                costs,
                t0,
                extra={
                    "fallback": True,
                    "profile_distance": profile_distance,
                    "structure_tolerance": float(structure_tolerance),
                    "model_path": model_path,
                },
            )

        tracker = helper.new_tracker()
        while True:
            fd = tracker.current_fdr()
            fr = tracker.current_fir()
            if fd >= tau_d and fr >= tau_i:
                break
            feats_now = helper.feature_matrix(tracker.selected)
            feats_norm = (feats_now - mu) / (sd + 1e-8)
            scores = net.predict(feats_norm)
            scores[tracker.selected == 1] = -1e12
            # 带预算的选择
            order = np.argsort(-scores)
            added = False
            for j in order:
                if tracker.selected[j] == 1:
                    continue
                if budget is not None and float(budget) > 0.0:
                    cur_cost = float(tracker.selected.astype(float) @ helper.costs)
                    if cur_cost + float(helper.costs[j]) > float(budget):
                        continue
                tracker.add(int(j))
                added = True
                break
            if not added:
                break

        selected = tracker.selected_mask()
        # 后向清理（预算约束）
        if selected.any():
            improved = True
            while improved:
                improved = False
                active = np.where(selected == 1)[0]
                for j in active:
                    trial = selected.copy()
                    trial[j] = 0
                    fd_trial, fr_trial, c_trial = helper.evaluate_mask(trial)
                    within_budget = (budget is None or float(budget) <= 0.0 or c_trial <= float(budget))
                    if fd_trial >= tau_d and fr_trial >= tau_i and within_budget:
                        selected = trial
                        improved = True

        return BaseAlgo._wrap_result(
            self.name,
            selected,
            D,
            probs,
            costs,
            t0,
            extra={
                "fallback": False,
                "profile_distance": float(profile_distance if np.isfinite(profile_distance) else 0.0),
                "structure_tolerance": float(structure_tolerance),
                "model_path": model_path,
            },
        )
