# -*- coding: utf-8 -*-
"""Neural MIP solver combining learned hints with CP-SAT."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper
from core.data_io import random_dataset
from core.solvers import solve_tp_mip_cp_sat


class _TinyMLP:
    """A minimal two-layer MLP implemented with NumPy for fast inference."""

    def __init__(self, in_dim: int, hidden: int = 32, lr: float = 1e-2, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden, dtype=float)
        self.W2 = rng.normal(0.0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1, dtype=float)
        self.lr = lr

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Hpre = X @ self.W1 + self.b1
        self.H = np.maximum(self.Hpre, 0.0)
        self.Y = self.H @ self.W2 + self.b2
        return self.Y

    def backward(self, gradY: np.ndarray) -> None:
        dW2 = self.H.T @ gradY
        db2 = gradY.sum(axis=0)
        dH = gradY @ self.W2.T
        dH[self.Hpre <= 0.0] = 0.0
        dW1 = self.X.T @ dH
        db1 = dH.sum(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit_mse(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, batch: int = 128) -> None:
        N = X.shape[0]
        idx = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                part = idx[s:s + batch]
                pred = self.forward(X[part])
                err = pred - y[part].reshape(-1, 1)
                self.backward(2.0 * err / max(1, part.size))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).reshape(-1)


@dataclass
class _CachedModel:
    net: _TinyMLP
    mean: np.ndarray
    std: np.ndarray


class NeuralMIPAlgo(BaseAlgo):
    """Implements Neural Diving hints + CP-SAT solving as described in the guide."""

    name = "NeuralMIP"

    def __init__(self):
        # Cache trained models keyed by problem shape & thresholds to avoid retraining.
        self._model_cache: Dict[Tuple[int, int, float, float], _CachedModel] = {}

    # ------------------------------------------------------------------
    # Model training utilities
    # ------------------------------------------------------------------
    def _collect_teacher_pairs(
        self,
        m: int,
        n: int,
        tau_d: float,
        tau_i: float,
        density: float,
        num_inst: int,
        time_limit: float,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        base_seed = 0 if seed is None else int(seed)
        for k in range(num_inst):
            ds = random_dataset(m, n, density=density, seed=base_seed + k)
            sol = solve_tp_mip_cp_sat(
                ds.D,
                ds.fault_probs,
                ds.test_costs,
                tau_d,
                tau_i,
                time_limit_s=time_limit,
                x_hint=None,
                log=False,
                num_workers=1,
            )
            helper = BinaryMetricHelper(ds.D, ds.fault_probs, ds.test_costs)
            feats = helper.feature_matrix(np.zeros(n, dtype=np.uint8))
            X_list.append(feats.astype(float))
            y_list.append(sol["selected"].astype(float).reshape(-1, 1))
        X = np.vstack(X_list)
        y = np.vstack(y_list).reshape(-1)
        return X, y

    def _train_model(
        self, X: np.ndarray, y: np.ndarray, hidden: int, epochs: int, seed: Optional[int]
    ) -> _CachedModel:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mu) / sd
        net = _TinyMLP(in_dim=X.shape[1], hidden=hidden, lr=1e-2, seed=seed)
        net.fit_mse(Xn, y, epochs=epochs, batch=128)
        return _CachedModel(net=net, mean=mu, std=sd)

    def _ensure_model(
        self,
        m: int,
        n: int,
        tau_d: float,
        tau_i: float,
        density: float,
        train_instances: int,
        teacher_time_limit: float,
        hidden: int,
        epochs: int,
        seed: Optional[int],
        use_cache: bool,
    ) -> _CachedModel:
        key = (m, n, round(tau_d, 3), round(tau_i, 3))
        if use_cache and key in self._model_cache:
            return self._model_cache[key]

        X, y = self._collect_teacher_pairs(
            m,
            n,
            tau_d,
            tau_i,
            density=density,
            num_inst=train_instances,
            time_limit=teacher_time_limit,
            seed=seed,
        )
        cached = self._train_model(X, y, hidden=hidden, epochs=epochs, seed=seed)
        if use_cache:
            self._model_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Hint prediction
    # ------------------------------------------------------------------
    @staticmethod
    def _predict_hint(
        cached: _CachedModel,
        helper: BinaryMetricHelper,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        feats = helper.feature_matrix(np.zeros(helper.n, dtype=np.uint8))
        feats_norm = (feats - cached.mean) / cached.std
        scores = cached.net.predict(feats_norm)
        prob = 1.0 / (1.0 + np.exp(-scores))
        hint = (prob >= threshold).astype(int)
        if int(hint.sum()) == 0:
            # fallback: choose the column with best (coverage+separation)/cost ratio
            cov = feats[:, 2]
            sep = feats[:, 4]
            cost = np.maximum(feats[:, 1], 1e-6)
            j = int(np.argmax((cov + sep) / cost))
            hint[j] = 1
        return hint.astype(int), prob

    @staticmethod
    def _enforce_feasible_hint(
        prob: np.ndarray,
        helper: BinaryMetricHelper,
        tau_d: float,
        tau_i: float,
    ) -> Optional[np.ndarray]:
        ratio = prob / (helper.costs + 1e-6)
        order = np.argsort(ratio)[::-1]
        tracker = helper.new_tracker()
        for idx in order:
            tracker.add(int(idx))
            if tracker.current_fdr() >= tau_d and tracker.current_fir() >= tau_i:
                return tracker.selected_mask()
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        D: np.ndarray,
        probs: np.ndarray,
        costs: np.ndarray,
        tau_d: float,
        tau_i: float,
        seed: Optional[int] = None,
        train_instances: int = 6,
        teacher_time_limit: float = 0.6,
        hidden: int = 32,
        epochs: int = 250,
        hint_threshold: float = 0.55,
        solver_time_limit: float = 1.0,
        density_override: Optional[float] = None,
        use_cache: bool = True,
    ) -> "AlgoResult":
        t0 = time.perf_counter()
        helper = BinaryMetricHelper(D, probs, costs)
        density = float(D.mean()) if density_override is None else float(density_override)

        cached = self._ensure_model(
            helper.m,
            helper.n,
            tau_d,
            tau_i,
            density=density,
            train_instances=train_instances,
            teacher_time_limit=teacher_time_limit,
            hidden=hidden,
            epochs=epochs,
            seed=seed,
            use_cache=use_cache,
        )

        hint, prob = self._predict_hint(cached, helper, threshold=hint_threshold)

        fd_hint, fi_hint, _ = helper.evaluate_mask(hint)
        if fd_hint < tau_d or fi_hint < tau_i:
            adjusted = self._enforce_feasible_hint(prob, helper, tau_d, tau_i)
            if adjusted is not None:
                hint = adjusted
                fd_hint, fi_hint, _ = helper.evaluate_mask(hint)

        if fd_hint >= tau_d and fi_hint >= tau_i:
            extra = {
                "hint_density": float(hint.mean()),
                "hint_positive": int(hint.sum()),
                "solve_time": 0.0,
                "hint_prob_mean": float(prob.mean()),
                "hint_only": True,
            }
            return BaseAlgo._wrap_result(
                self.name,
                hint,
                D,
                probs,
                costs,
                t0,
                extra=extra,
            )

        solve = solve_tp_mip_cp_sat(
            D,
            probs,
            costs,
            tau_d,
            tau_i,
            time_limit_s=solver_time_limit,
            x_hint=hint,
            log=False,
            num_workers=1,
        )

        if not solve["feasible"]:
            # fallback to greedy heuristic if CP-SAT failed within the limit
            from core.algos.greedy import GreedyAlgo

            greedy_res = GreedyAlgo().run(D, probs, costs, tau_d, tau_i, seed=seed)
            greedy_res.extra.setdefault("fallback", "greedy")
            greedy_res.extra.setdefault("hint_density", float(hint.mean()))
            return greedy_res

        extra = {
            "hint_density": float(hint.mean()),
            "hint_positive": int(hint.sum()),
            "solve_time": float(solve["solve_time"]),
            "hint_prob_mean": float(prob.mean()),
        }

        return BaseAlgo._wrap_result(
            self.name,
            solve["selected"],
            D,
            probs,
            costs,
            t0,
            extra=extra,
        )

