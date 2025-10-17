# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper

# 简单的模型缓存（同一 D 或相近特征分布可复用）
_MODEL_CACHE = {}


class TinyMLP:
    """纯 numpy 的两层 MLP: in -> hidden(ReLU) -> out(线性)"""

    def __init__(self, in_dim, hidden=16, lr=1e-2, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.Hpre = X.dot(self.W1) + self.b1
        self.H = np.maximum(0.0, self.Hpre)
        self.Y = self.H.dot(self.W2) + self.b2
        return self.Y

    def backward(self, gradY):
        dW2 = self.H.T.dot(gradY)
        db2 = gradY.sum(axis=0)
        dH = gradY.dot(self.W2.T)
        dH[self.Hpre <= 0] = 0.0
        dW1 = self.X.T.dot(dH)
        db1 = dH.sum(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit_mse(self, X, y, epochs=200, batch=64):
        N = X.shape[0]
        for _ in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                part = idx[s:s + batch]
                ypred = self.forward(X[part])
                err = ypred - y[part].reshape(-1, 1)
                self.backward(2.0 * err / max(1, part.size))

    def predict(self, X):
        return self.forward(X).reshape(-1)


class NNGuidedAlgo(BaseAlgo):
    name = "NN-Guided"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            synth_samples=150, epochs=200, hidden=16, w_fdr=0.5, w_fir=0.5,
            budget: float | None = None, use_cache: bool = True) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n

        # 训练或复用缓存的模型
        key = (n, m_hash(D))
        cached = False
        if use_cache and key in _MODEL_CACHE:
            net, scaler_mean, scaler_std = _MODEL_CACHE[key]
            cached = True
        else:
            X_list = []
            y_list = []
            tracker = helper.new_tracker()
            for _ in range(synth_samples):
                mask = (rng.random(n) < 0.4).astype(np.uint8)
                tracker.build_from_mask(mask)
                feats = helper.feature_matrix(tracker.selected)
                fdr_gain, fir_gain = tracker.gain_vectors()
                gains = w_fdr * fdr_gain + w_fir * fir_gain
                available = tracker.selected == 0
                if not np.any(available):
                    continue
                denom = np.maximum(helper.costs[available], 1e-12)
                X_list.append(feats[available])
                y_list.append(gains[available] / denom)

            if not X_list:
                from core.algos.greedy import GreedyAlgo
                return GreedyAlgo().run(D, probs, costs, tau_d, tau_i, seed=seed)

            X = np.vstack(X_list)
            y = np.concatenate(y_list)

            scaler_mean = X.mean(axis=0, keepdims=True)
            scaler_std = X.std(axis=0, keepdims=True) + 1e-8
            Xn = (X - scaler_mean) / scaler_std

            net = TinyMLP(in_dim=X.shape[1], hidden=hidden, lr=1e-2, seed=seed)
            net.fit_mse(Xn, y, epochs=epochs, batch=64)
            if use_cache:
                _MODEL_CACHE[key] = (net, scaler_mean, scaler_std)

        tracker = helper.new_tracker()
        while True:
            fd = tracker.current_fdr()
            fr = tracker.current_fir()
            if fd >= tau_d and fr >= tau_i:
                break
            feats_now = helper.feature_matrix(tracker.selected)
            feats_norm = (feats_now - scaler_mean) / scaler_std
            scores = net.predict(feats_norm)
            scores[tracker.selected == 1] = -1e12
            # 考虑预算的选择策略：从高分到低分尝试可行的列
            candidate_idx = np.argsort(-scores)
            added = False
            for j in candidate_idx:
                if tracker.selected[j] == 1:
                    continue
                if scores[j] <= 0 and not added:
                    # 兜底：选性价比最高的列
                    remaining = np.where(tracker.selected == 0)[0]
                    if remaining.size == 0:
                        break
                    j = int(remaining[np.argmin(helper.costs[remaining])])
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
        # 后向清理：保证在预算内，同时不破坏阈值
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

        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "cached": bool(cached),
        })


def m_hash(D: np.ndarray) -> int:
    # 对 D 的稀疏结构做一个快速哈希（仅用于缓存键）
    D = (np.asarray(D) > 0).astype(np.uint8)
    # 取部分哈希避免大数组代价
    return int(hash(D.tobytes()[: min(D.size, 8192)]))
