# -*- coding: utf-8 -*-
import os
import time
import numpy as np

from core.algos.base import BaseAlgo
from experiments.features import per_test_features
from experiments.models import load_tinymlp


class NNHintMIPOfflineAlgo(BaseAlgo):
    name = "NN-MIP_Offline"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            model_path: str = os.path.join("data", "models", "nn_mip_offline.npz"),
            time_limit_s: float = 5.0, hint_th: float = 0.5,
            budget: float | None = None, num_workers: int = 1, use_callback: bool = False) -> "AlgoResult":
        t0 = time.perf_counter()
        m, n = D.shape
        if not os.path.exists(model_path):
            # 回退到在线版本
            from core.algos.nn_mip import NNHintMIPAlgo
            res = NNHintMIPAlgo().run(D, probs, costs, tau_d, tau_i, seed=seed,
                                      time_limit_s=time_limit_s, hint_th=hint_th,
                                      budget=budget, num_workers=num_workers, use_callback=use_callback)
            return BaseAlgo._wrap_result(self.name, res.selected, D, probs, costs, t0, extra={"fallback": True})

        net, mu, sd = load_tinymlp(model_path)

        ctx = np.zeros(n, dtype=int)
        feats = per_test_features(D, probs, costs, ctx)
        scores = net.predict((feats - mu) / (sd + 1e-8))
        prob = 1.0 / (1.0 + np.exp(-scores))
        hint = (prob >= float(hint_th)).astype(int)
        if hint.sum() == 0:
            j = int(np.argmax(scores))
            hint[j] = 1

        try:
            from experiments.ilp_cp_sat import solve_tp_mip_cp_sat
        except Exception as e:
            raise RuntimeError(
                "NN-MIP_Offline 需要安装 OR-Tools (pip install ortools)。"
            ) from e

        sol = solve_tp_mip_cp_sat(D, probs, costs, tau_d, tau_i,
                                  time_limit_s=time_limit_s, x_hint=hint, log=False,
                                  budget=budget, num_workers=num_workers, use_callback=use_callback)
        selected = sol.get("selected")
        if selected is None:
            selected = np.zeros(n, dtype=int)
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={"fallback": False})

