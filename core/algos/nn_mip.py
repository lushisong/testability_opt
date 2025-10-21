# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import structure_distance, structure_profile
from experiments.features import per_test_features

try:
    from core.algos.nn_guided import TinyMLP
except Exception:  # pragma: no cover - fallback rarely used
    class TinyMLP:  # type: ignore[misc]
        def __init__(self, in_dim, hidden=16, lr=1e-2, seed=None):
            rng = np.random.default_rng(seed)
            self.W = rng.normal(0, 0.1, size=(in_dim, 1))
            self.b = np.zeros(1)
            self.lr = lr

        def forward(self, X):
            return X @ self.W + self.b

        def fit_mse(self, X, y, epochs=50, batch=128):
            N = X.shape[0]
            for _ in range(epochs):
                idx = np.arange(N)
                np.random.shuffle(idx)
                for s in range(0, N, batch):
                    part = idx[s:s+batch]
                    pred = self.forward(X[part])
                    err = pred.reshape(-1) - y[part]
                    gW = (X[part].T @ err.reshape(-1, 1)) * (2.0 / max(1, part.size))
                    gb = np.array([2.0 * float(err.mean())])
                    self.W -= self.lr * gW
                    self.b -= self.lr * gb

        def predict(self, X):
            return self.forward(X).reshape(-1)


@dataclass
class CachedHintModel:
    net: "TinyMLP"
    mu: np.ndarray
    sd: np.ndarray
    profile: Dict[str, object]
    train_time: float
    samples: int
    epochs: int
    created_at: float
    reuse_count: int = 0


_MIP_MODEL_CACHE: list[CachedHintModel] = []
_MIP_CACHE_STATS = {"queries": 0, "hits": 0}
_MIP_CACHE_CAPACITY = 8


def _locate_cached_mip_model(profile: Dict[str, object], tol: float) -> Tuple[Optional[CachedHintModel], float]:
    best_entry: Optional[CachedHintModel] = None
    best_dist = float("inf")
    for entry in _MIP_MODEL_CACHE:
        if entry.profile["shape"] != profile["shape"]:
            continue
        dist = structure_distance(entry.profile, profile)
        if dist < best_dist:
            best_entry = entry
            best_dist = dist
    if best_entry is None or best_dist > tol:
        return None, best_dist
    return best_entry, best_dist


def _store_cached_mip_model(entry: CachedHintModel) -> None:
    _MIP_MODEL_CACHE.append(entry)
    if len(_MIP_MODEL_CACHE) > _MIP_CACHE_CAPACITY:
        _MIP_MODEL_CACHE.sort(key=lambda item: (item.reuse_count, item.created_at))
        _MIP_MODEL_CACHE.pop(0)


class NNHintMIPAlgo(BaseAlgo):
    """Neural hinting + CP-SAT pipeline with structural aware caching."""

    name = "NN-MIP"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None,
            time_limit_s: float = 5.0,
            hidden: int = 32, epochs: int = 120, hint_th: float = 0.5,
            budget: float | None = None, num_workers: int = 1, use_callback: bool = False,
            use_cache: bool = True, structure_tolerance: float = 0.05) -> "AlgoResult":
        t0 = time.perf_counter()
        _, n = D.shape
        profile = structure_profile(D)

        ctx = np.zeros(n, dtype=int)
        feats = per_test_features(D, probs, costs, ctx)

        cached = False
        cache_distance = float("inf")
        train_time_run = 0.0
        train_samples = 0
        reuse_count = 0

        _MIP_CACHE_STATS["queries"] += 1
        if use_cache:
            entry, cache_distance = _locate_cached_mip_model(profile, structure_tolerance)
            if entry is not None:
                cached = True
                _MIP_CACHE_STATS["hits"] += 1
                entry.reuse_count += 1
                reuse_count = entry.reuse_count
                net = entry.net
                mu = entry.mu
                sd = entry.sd
            else:
                net = None  # type: ignore[assignment]
                mu = np.zeros((1, feats.shape[1]))
                sd = np.ones((1, feats.shape[1]))
        else:
            net = None  # type: ignore[assignment]
            mu = np.zeros((1, feats.shape[1]))
            sd = np.ones((1, feats.shape[1]))

        if not cached:
            num = feats[:, 2] + feats[:, 4]
            den = feats[:, 1] + 1e-6
            y = num / den

            mu = feats.mean(axis=0, keepdims=True)
            sd = feats.std(axis=0, keepdims=True) + 1e-8
            Xn = (feats - mu) / sd

            net = TinyMLP(in_dim=Xn.shape[1], hidden=hidden, lr=1e-2, seed=seed)
            t_train = time.perf_counter()
            net.fit_mse(Xn, y, epochs=epochs, batch=128)
            train_time_run = time.perf_counter() - t_train
            train_samples = int(Xn.shape[0])
            if use_cache:
                entry = CachedHintModel(
                    net=net,
                    mu=mu,
                    sd=sd,
                    profile=profile,
                    train_time=train_time_run,
                    samples=train_samples,
                    epochs=int(epochs),
                    created_at=time.time(),
                )
                _store_cached_mip_model(entry)
                reuse_count = entry.reuse_count
        else:
            num = feats[:, 2] + feats[:, 4]
            den = feats[:, 1] + 1e-6
            y = num / den

        Xn = (feats - mu) / sd
        scores = net.predict(Xn)
        prob = 1.0 / (1.0 + np.exp(-scores))
        hint = (prob >= float(hint_th)).astype(int)
        if hint.sum() == 0:
            j = int(np.argmax(y))
            hint[j] = 1

        try:
            from experiments.ilp_cp_sat import solve_tp_mip_cp_sat
        except Exception as e:  # pragma: no cover - requires ortools
            raise RuntimeError(
                "NN-MIP 需要安装 OR-Tools (pip install ortools)。\n"
                "请安装依赖或改用 Greedy/Firefly/PSO/NN-Guided。"
            ) from e

        sol = solve_tp_mip_cp_sat(
            D,
            probs,
            costs,
            tau_d,
            tau_i,
            time_limit_s=time_limit_s,
            x_hint=hint,
            log=False,
            budget=budget,
            num_workers=num_workers,
            use_callback=use_callback,
        )

        selected = sol.get("selected")
        if selected is None:
            selected = np.zeros(n, dtype=int)

        cache_hits = _MIP_CACHE_STATS["hits"] if use_cache else 0
        cache_queries = _MIP_CACHE_STATS["queries"] if use_cache else 0

        return BaseAlgo._wrap_result(
            self.name,
            selected,
            D,
            probs,
            costs,
            t0,
            extra={
                "hint_cnt": int(hint.sum()),
                "cp_status": int(sol.get("status", -1)),
                "feasible": bool(sol.get("feasible", False)),
                "cached": bool(cached),
                "train_time_sec": float(train_time_run),
                "train_samples": int(train_samples),
                "profile_distance": float(0.0 if not cached else cache_distance),
                "structure_tolerance": float(structure_tolerance),
                "reuse_count": int(reuse_count),
                "cache_hits": int(cache_hits),
                "cache_queries": int(cache_queries),
            },
        )
