# -*- coding: utf-8 -*-
import time
import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper


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
            synth_samples=150, epochs=200, hidden=16, w_fdr=0.5, w_fir=0.5) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n

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

        tracker.reset()
        while True:
            fd = tracker.current_fdr()
            fr = tracker.current_fir()
            if fd >= tau_d and fr >= tau_i:
                break
            feats_now = helper.feature_matrix(tracker.selected)
            feats_norm = (feats_now - scaler_mean) / scaler_std
            scores = net.predict(feats_norm)
            scores[tracker.selected == 1] = -1e12
            j = int(np.argmax(scores))
            if scores[j] <= 0:
                remaining = np.where(tracker.selected == 0)[0]
                if remaining.size == 0:
                    break
                j = int(remaining[np.argmin(helper.costs[remaining])])
            tracker.add(j)

        selected = tracker.selected_mask()
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={
            "samples": len(y),
        })
