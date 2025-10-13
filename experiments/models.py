# -*- coding: utf-8 -*-
import numpy as np

class TinyMLP:
    def __init__(self, in_dim, hidden=32, lr=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.Hpre = X @ self.W1 + self.b1
        self.H = np.maximum(0.0, self.Hpre)
        self.Y = self.H @ self.W2 + self.b2
        return self.Y

    def backward(self, gradY):
        dW2 = self.H.T @ gradY
        db2 = gradY.sum(axis=0)
        dH = gradY @ self.W2.T
        dH[self.Hpre <= 0] = 0.0
        dW1 = self.X.T @ dH
        db1 = dH.sum(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit_mse(self, X, y, epochs=200, batch=128):
        N = X.shape[0]
        for ep in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                part = idx[s:s+batch]
                pred = self.forward(X[part])
                err = pred - y[part].reshape(-1, 1)
                self.backward(2.0 * err / max(1, part.size))

    def predict(self, X):
        return self.forward(X).reshape(-1)

