# -*- coding: utf-8 -*-
import time
import numpy as np
from core.algos.base import BaseAlgo
from core.metrics import fdr, fir, cost, marginal_gain

class GreedyAlgo(BaseAlgo):
    name = "Greedy"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            w_fdr=0.5, w_fir=0.5, backward_cleanup=True) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        m, n = D.shape
        selected = np.zeros(n, dtype=int)

        # 迭代选择，直到满足阈值或无改进
        while True:
            fd = fdr(selected, D, probs)
            fr = fir(selected, D, probs)
            if fd >= tau_d and fr >= tau_i:
                break
            # 选择单位代价收益最大的测试
            best_j, best_score = -1, -1.0
            for j in range(n):
                if selected[j] == 1:
                    continue
                dfd, dfr, sc = marginal_gain(selected, j, D, probs, w_fdr, w_fir)
                denom = max(costs[j], 1e-6)
                score = sc / denom
                if score > best_score + 1e-12:
                    best_score = score
                    best_j = j
            if best_j < 0:
                # 无可提升，随机加入一个成本最低的未选测试以尝试突破
                unpicked = np.where(selected == 0)[0]
                if unpicked.size == 0:
                    break
                best_j = unpicked[np.argmin(costs[unpicked])]
            selected[best_j] = 1

        # 后向删除冗余
        if backward_cleanup:
            improved = True
            while improved:
                improved = False
                idx = np.where(selected == 1)[0]
                for j in list(idx):
                    selected[j] = 0
                    if fdr(selected, D, probs) >= tau_d and fir(selected, D, probs) >= tau_i:
                        improved = True
                        # keep removed
                    else:
                        selected[j] = 1

        from core.algos.base import BaseAlgo
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={})
