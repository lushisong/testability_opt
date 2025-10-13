# -*- coding: utf-8 -*-
import time
import numpy as np
from core.algos.base import BaseAlgo
from core.metrics import fdr, fir, marginal_gain

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
        # X: (N, in_dim)
        self.X = X
        self.Hpre = X.dot(self.W1) + self.b1
        self.H = np.maximum(0.0, self.Hpre)
        self.Y = self.H.dot(self.W2) + self.b2  # (N, 1)
        return self.Y

    def backward(self, gradY):
        # gradY: (N, 1)
        dW2 = self.H.T.dot(gradY)
        db2 = gradY.sum(axis=0)
        dH = gradY.dot(self.W2.T)
        dH[self.Hpre <= 0] = 0.0
        dW1 = self.X.T.dot(dH)
        db1 = dH.sum(axis=0)
        # 更新
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit_mse(self, X, y, epochs=200, batch=64):
        N = X.shape[0]
        for ep in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                part = idx[s:s+batch]
                ypred = self.forward(X[part])
                err = ypred - y[part].reshape(-1, 1)
                self.backward(2.0 * err / part.size)

    def predict(self, X):
        return self.forward(X).reshape(-1)

class NNGuidedAlgo(BaseAlgo):
    name = "NN-Guided"

    def _features(self, D, probs, costs, selected):
        """为每个测试提取上下文相关的特征"""
        m, n = D.shape
        sel_cols = np.where(selected == 1)[0]
        covered = (D[:, sel_cols].sum(axis=1) > 0) if sel_cols.size > 0 else np.zeros(m, dtype=bool)

        feats = []
        for j in range(n):
            col = D[:, j]
            # 覆盖提升
            new_cover = (~covered) & (col == 1)
            w_cov = float((probs[new_cover]).sum())
            cnt_cov = int(new_cover.sum())
            # 与当前选择的列对比会带来多少成对区分提升（粗略近似）
            # 用列与当前签名的差异度估计
            if sel_cols.size == 0:
                sep_gain = float((probs[col == 1]).sum())  # 初始用覆盖权重近似
            else:
                sig = D[:, sel_cols].dot(1 << np.arange(sel_cols.size, dtype=np.int64))
                sig_new = D[:, np.r_[sel_cols, j]].dot(1 << np.arange(sel_cols.size + 1, dtype=np.int64))
                # 可区分对数近似：签名变化的故障数量指标
                sep_gain = float(np.mean(sig != sig_new))
            feats.append([1.0, costs[j], w_cov, cnt_cov, sep_gain])
        return np.array(feats, dtype=float)  # shape (n, 5)

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            synth_samples=150, epochs=200, hidden=16, w_fdr=0.5, w_fir=0.5) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        m, n = D.shape

        # 合成训练数据：随机上下文下的边际收益标签
        X_list, y_list = [], []
        for _ in range(synth_samples):
            selected = (rng.random(n) < 0.4).astype(int)
            for j in range(n):
                if selected[j] == 1:
                    continue
                dfd, dfr, sc = marginal_gain(selected, j, D, probs, w_fdr, w_fir)
                feats = self._features(D, probs, costs, selected)[j]
                X_list.append(feats)
                y_list.append(sc / max(costs[j], 1e-6))
        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        if X.shape[0] == 0:
            # 退化情形：直接退回贪心
            from core.algos.greedy import GreedyAlgo
            return GreedyAlgo().run(D, probs, costs, tau_d, tau_i, seed=seed)

        scaler_mean = X.mean(axis=0, keepdims=True)
        scaler_std = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - scaler_mean) / scaler_std

        net = TinyMLP(in_dim=X.shape[1], hidden=hidden, lr=1e-2, seed=seed)
        net.fit_mse(Xn, y, epochs=epochs, batch=64)

        # 用学习到的评分 + 贪心选择（每步重新提取上下文特征并评分）
        selected = np.zeros(n, dtype=int)
        while True:
            fd = fdr(selected, D, probs)
            fr = fir(selected, D, probs)
            if fd >= tau_d and fr >= tau_i:
                break
            feats_now = self._features(D, probs, costs, selected)
            scores = net.predict((feats_now - scaler_mean) / scaler_std)
            scores[selected == 1] = -1e9
            j = int(np.argmax(scores))
            if scores[j] <= 0:
                # 无正向收益，回退为按单位代价覆盖选
                cand = np.where(selected == 0)[0]
                if cand.size == 0:
                    break
                j = int(cand[np.argmin(costs[cand])])
            selected[j] = 1

        from core.algos.base import BaseAlgo
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra={"samples": X.shape[0]})
