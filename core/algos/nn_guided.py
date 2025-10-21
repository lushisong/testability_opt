# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import (
    BinaryMetricHelper,
    structure_distance,
    structure_profile,
)


@dataclass
class CachedNNModel:
    net: "TinyMLP"
    mu: np.ndarray
    sd: np.ndarray
    profile: Dict[str, object]
    train_time: float
    samples: int
    epochs: int
    created_at: float
    reuse_count: int = 0


_MODEL_CACHE: list[CachedNNModel] = []
_CACHE_STATS = {"queries": 0, "hits": 0}
_CACHE_CAPACITY = 8


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


def _locate_cached_model(profile: Dict[str, object], tol: float) -> Tuple[Optional[CachedNNModel], float]:
    best_entry: Optional[CachedNNModel] = None
    best_dist = float("inf")
    for entry in _MODEL_CACHE:
        if entry.profile["shape"] != profile["shape"]:
            continue
        dist = structure_distance(entry.profile, profile)
        if dist < best_dist:
            best_entry = entry
            best_dist = dist
    if best_entry is None or best_dist > tol:
        return None, best_dist
    return best_entry, best_dist


def _store_cached_model(entry: CachedNNModel) -> None:
    _MODEL_CACHE.append(entry)
    if len(_MODEL_CACHE) > _CACHE_CAPACITY:
        _MODEL_CACHE.sort(key=lambda item: (item.reuse_count, item.created_at))
        _MODEL_CACHE.pop(0)


class NNGuidedAlgo(BaseAlgo):
    name = "NN-Guided"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            synth_samples=150, epochs=200, hidden=16, w_fdr=0.5, w_fir=0.5,
            budget: float | None = None, use_cache: bool = True,
            structure_tolerance: float = 0.05) -> "AlgoResult":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        helper = BinaryMetricHelper(D, probs, costs)
        n = helper.n
        profile = structure_profile(D)

        cached = False
        cache_distance = float("inf")
        train_time_run = 0.0
        train_samples = 0
        reuse_count = 0
        net: TinyMLP
        scaler_mean: np.ndarray
        scaler_std: np.ndarray

        _CACHE_STATS["queries"] += 1
        if use_cache:
            entry, cache_distance = _locate_cached_model(profile, structure_tolerance)
            if entry is not None:
                cached = True
                _CACHE_STATS["hits"] += 1
                entry.reuse_count += 1
                reuse_count = entry.reuse_count
                net = entry.net
                scaler_mean = entry.mu
                scaler_std = entry.sd
            else:
                net = None  # type: ignore[assignment]
                scaler_mean = np.zeros((1, 0))
                scaler_std = np.ones((1, 0))
        else:
            net = None  # type: ignore[assignment]
            scaler_mean = np.zeros((1, 0))
            scaler_std = np.ones((1, 0))

        if not cached:
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
            t_train = time.perf_counter()
            net.fit_mse(Xn, y, epochs=epochs, batch=64)
            train_time_run = time.perf_counter() - t_train
            train_samples = int(X.shape[0])
            if use_cache:
                entry = CachedNNModel(
                    net=net,
                    mu=scaler_mean,
                    sd=scaler_std,
                    profile=profile,
                    train_time=train_time_run,
                    samples=train_samples,
                    epochs=int(epochs),
                    created_at=time.time(),
                )
                _store_cached_model(entry)
                reuse_count = entry.reuse_count
        else:
            train_time_run = 0.0
            train_samples = 0

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
            candidate_idx = np.argsort(-scores)
            added = False
            for j in candidate_idx:
                if tracker.selected[j] == 1:
                    continue
                if scores[j] <= 0 and not added:
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

        cache_hits = _CACHE_STATS["hits"] if use_cache else 0
        cache_queries = _CACHE_STATS["queries"] if use_cache else 0
        extra = {
            "cached": bool(cached),
            "train_time_sec": float(train_time_run),
            "train_samples": int(train_samples),
            "profile_distance": float(0.0 if not cached else cache_distance),
            "structure_tolerance": float(structure_tolerance),
            "reuse_count": int(reuse_count),
            "cache_hits": int(cache_hits),
            "cache_queries": int(cache_queries),
        }
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra=extra)
