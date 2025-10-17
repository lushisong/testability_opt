# -*- coding: utf-8 -*-
"""Neural branching policy integrating learned GNN scores with CP-SAT."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core.algos.base import BaseAlgo
from core.algos.utils import BinaryMetricHelper, test_adjacency_matrix
from core.solvers import BranchingStrategy, solve_tp_mip_cp_sat

try:  # torch is required when using the learned policy
    import torch
except Exception as exc:  # pragma: no cover - defensive
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from experiments.models import load_branching_gcn


@dataclass
class _LoadedModel:
    model: any
    meta: dict


class NeuralBranchingAlgo(BaseAlgo):
    name = "NeuralBranching"

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self._cached: Optional[_LoadedModel] = None

    # ------------------------------------------------------------------
    # Model utilities
    # ------------------------------------------------------------------
    def _ensure_model(self) -> _LoadedModel:
        if torch is None:
            raise RuntimeError("torch is required for NeuralBranching") from _TORCH_IMPORT_ERROR
        if self._cached is None:
            if self.model_path is None:
                raise ValueError("model_path must be provided for learned branching")
            model, meta = load_branching_gcn(str(self.model_path), map_location=self.device)
            model.to(self.device)
            self._cached = _LoadedModel(model=model, meta=meta)
        return self._cached

    # ------------------------------------------------------------------
    # Ordering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_order(helper: BinaryMetricHelper) -> np.ndarray:
        tracker = helper.new_tracker()
        tracker.reset()
        feats = helper.feature_matrix(tracker.selected)
        gains = feats[:, 2] + feats[:, 4]
        order = np.argsort(-(gains / (helper.costs + 1e-6)))
        return order.astype(int)

    def _learned_order(self, helper: BinaryMetricHelper) -> Tuple[np.ndarray, np.ndarray]:
        loaded = self._ensure_model()
        adjacency = test_adjacency_matrix(helper.D_bool)
        selected = np.zeros(helper.n, dtype=np.float32)
        feats = helper.feature_matrix(selected)
        with torch.no_grad():
            logits = loaded.model(
                torch.from_numpy(feats.astype(np.float32)).to(self.device),
                torch.from_numpy(adjacency.astype(np.float32)).to(self.device),
                torch.from_numpy(selected.astype(np.float32)).to(self.device),
            )
        order = torch.argsort(logits, descending=True)
        return order.cpu().numpy().astype(int), logits.cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        D,
        probs,
        costs,
        tau_d,
        tau_i,
        seed=None,
        time_limit: float = 10.0,
        branching_mode: str = "learned",
        fallback: bool = True,
        num_workers: int = 1,
        fallback_time: Optional[float] = None,
    ):
        t0 = time.perf_counter()
        helper = BinaryMetricHelper(D, probs, costs)
        adjacency = test_adjacency_matrix(helper.D_bool)

        logits = None
        if branching_mode == "default":
            order = self._default_order(helper)
        elif branching_mode == "learned":
            order, logits = self._learned_order(helper)
        else:
            raise ValueError("branching_mode must be 'default' or 'learned'")

        branching = BranchingStrategy(order=order.tolist()) if order is not None else None
        sol = solve_tp_mip_cp_sat(
            helper.D_bool,
            helper.probs,
            helper.costs,
            tau_d,
            tau_i,
            time_limit_s=time_limit,
            num_workers=num_workers,
            branching=branching,
        )

        used_fallback = False
        if fallback and (not sol.get("feasible", False)) and branching_mode == "learned":
            sol_default = solve_tp_mip_cp_sat(
                helper.D_bool,
                helper.probs,
                helper.costs,
                tau_d,
                tau_i,
                time_limit_s=fallback_time or time_limit,
                num_workers=num_workers,
                branching=None,
            )
            if sol_default.get("feasible", False):
                sol = sol_default
                used_fallback = True

        selected = sol["selected"]
        extra = {
            "solver": sol,
            "branching_mode": branching_mode,
            "used_fallback": used_fallback,
            "logits": logits,
            "adjacency_shape": adjacency.shape,
        }
        return BaseAlgo._wrap_result(self.name, selected, D, probs, costs, t0, extra=extra)
