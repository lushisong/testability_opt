# -*- coding: utf-8 -*-
"""Utility helpers shared by the optimization algorithms."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class _PairwiseData:
    diff_matrix: np.ndarray  # shape (n, pair_count), uint8
    pair_weights: np.ndarray  # shape (pair_count,)
    total_weight: float


def structure_profile(D: np.ndarray, sample: int = 16) -> Dict[str, object]:
    """Generate a lightweight structural signature for caching and similarity checks."""
    D_bin = (np.asarray(D, dtype=np.uint8) > 0).astype(np.uint8)
    m, n = D_bin.shape
    nnz = int(D_bin.sum())
    row_sums = D_bin.sum(axis=1).astype(int)
    col_sums = D_bin.sum(axis=0).astype(int)
    top_rows = tuple(np.sort(row_sums)[-min(sample, m):].tolist())
    top_cols = tuple(np.sort(col_sums)[-min(sample, n):].tolist())
    return {
        "shape": (int(m), int(n)),
        "nnz": nnz,
        "row_top": top_rows,
        "col_top": top_cols,
    }


def structure_distance(a: Dict[str, object], b: Dict[str, object]) -> float:
    """Compute a normalized distance between two structural signatures."""
    shape_a = tuple(a["shape"])
    shape_b = tuple(b["shape"])
    if shape_a != shape_b:
        return math.inf
    m, n = shape_a
    total = max(1, m * n)
    nnz_a = float(a["nnz"])
    nnz_b = float(b["nnz"])
    nnz_gap = abs(nnz_a - nnz_b) / total

    def _l2_gap(key: str) -> float:
        arr_a = np.array(a[key], dtype=float)
        arr_b = np.array(b[key], dtype=float)
        if arr_a.size == 0 and arr_b.size == 0:
            return 0.0
        size = max(arr_a.size, arr_b.size)
        if arr_a.size < size:
            arr_a = np.pad(arr_a, (size - arr_a.size, 0))
        if arr_b.size < size:
            arr_b = np.pad(arr_b, (size - arr_b.size, 0))
        denom = max(1.0, float(arr_a.sum() + arr_b.sum()))
        return float(np.linalg.norm(arr_a - arr_b, ord=1)) / denom

    row_gap = _l2_gap("row_top")
    col_gap = _l2_gap("col_top")
    return nnz_gap + 0.5 * (row_gap + col_gap)


class BinaryMetricHelper:
    """Pre-computes structures to evaluate binary selections efficiently."""

    def __init__(self, D: np.ndarray, probs: np.ndarray, costs: np.ndarray):
        self.D_bool = (np.asarray(D, dtype=np.uint8) > 0).astype(np.uint8)
        self.probs = np.asarray(probs, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.total_prob = float(self.probs.sum())
        self.m, self.n = self.D_bool.shape
        self._pairwise = self._build_pairwise()

    def _build_pairwise(self) -> _PairwiseData:
        if self.m < 2:
            diff_matrix = np.zeros((self.n, 0), dtype=np.uint8)
            pair_weights = np.zeros(0, dtype=float)
            total_weight = 0.0
        else:
            idx_i, idx_j = np.triu_indices(self.m, k=1)
            pair_weights = (self.probs[idx_i] * self.probs[idx_j]).astype(float)
            diff_matrix = (self.D_bool[idx_i] != self.D_bool[idx_j]).astype(np.uint8).T
            total_weight = float(pair_weights.sum())
        return _PairwiseData(diff_matrix=diff_matrix, pair_weights=pair_weights, total_weight=total_weight)

    # ------------------------------------------------------------------
    # Population evaluation helpers
    # ------------------------------------------------------------------
    def _ensure_2d(self, pop: np.ndarray) -> np.ndarray:
        pop_arr = np.asarray(pop, dtype=np.uint8)
        if pop_arr.ndim == 1:
            pop_arr = pop_arr.reshape(1, -1)
        return pop_arr

    def evaluate_population(self, pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pop_arr = self._ensure_2d(pop)
        cover_counts = pop_arr @ self.D_bool.T
        covered = cover_counts > 0
        if self.total_prob > 0.0:
            fdr_vals = (covered * self.probs).sum(axis=1) / self.total_prob
        else:
            fdr_vals = np.zeros(pop_arr.shape[0], dtype=float)

        if self._pairwise.diff_matrix.size > 0 and self._pairwise.total_weight > 0.0:
            diff_counts = pop_arr @ self._pairwise.diff_matrix
            distinguished = diff_counts > 0
            fir_vals = (distinguished * self._pairwise.pair_weights).sum(axis=1) / self._pairwise.total_weight
        else:
            fir_vals = np.zeros(pop_arr.shape[0], dtype=float)

        costs = pop_arr.astype(float) @ self.costs
        return fdr_vals, fir_vals, costs

    def penalized_objective(self, pop: np.ndarray, tau_d: float, tau_i: float, penalty: float,
                             budget: float | None = None) -> np.ndarray:
        fdr_vals, fir_vals, costs = self.evaluate_population(pop)
        gd = np.maximum(0.0, tau_d - fdr_vals)
        gi = np.maximum(0.0, tau_i - fir_vals)
        gb = np.maximum(0.0, costs - float(budget)) if budget is not None else 0.0
        return costs + penalty * (gd + gi + gb)

    def evaluate_mask(self, mask: np.ndarray) -> Tuple[float, float, float]:
        fdr_vals, fir_vals, costs = self.evaluate_population(mask)
        return float(fdr_vals[0]), float(fir_vals[0]), float(costs[0])

    # ------------------------------------------------------------------
    # Feature and marginal gain utilities
    # ------------------------------------------------------------------
    def feature_matrix(self, selected_mask: np.ndarray) -> np.ndarray:
        sel = np.asarray(selected_mask, dtype=bool)
        if sel.shape != (self.n,):
            raise ValueError("selected_mask must have shape (n,)")
        if sel.any():
            covered = (self.D_bool[:, sel].sum(axis=1) > 0)
            diff_counts = sel.astype(np.uint8) @ self._pairwise.diff_matrix
        else:
            covered = np.zeros(self.m, dtype=bool)
            diff_counts = np.zeros(self._pairwise.diff_matrix.shape[1], dtype=np.uint8)
        uncovered_probs = np.where(~covered, self.probs, 0.0)
        coverage_weight_gain = self.D_bool.T @ uncovered_probs
        coverage_count_gain = self.D_bool[~covered].sum(axis=0).astype(float) if (~covered).any() else np.zeros(self.n, dtype=float)
        if self._pairwise.diff_matrix.size > 0:
            remaining = np.where(diff_counts > 0, 0.0, self._pairwise.pair_weights)
            separation_gain = self._pairwise.diff_matrix @ remaining
        else:
            separation_gain = np.zeros(self.n, dtype=float)
        feats = np.vstack([
            np.ones(self.n, dtype=float),
            self.costs,
            coverage_weight_gain.astype(float),
            coverage_count_gain,
            separation_gain.astype(float),
        ]).T
        return feats

    def marginal_gains(self, selected_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tracker = SelectionTracker(self)
        tracker.build_from_mask(selected_mask)
        return tracker.gain_vectors()

    def new_tracker(self) -> "SelectionTracker":
        return SelectionTracker(self)


class SelectionTracker:
    """Tracks incremental FDR/FIR gains for a growing selection."""

    def __init__(self, helper: BinaryMetricHelper):
        self.helper = helper
        self.n = helper.n
        self.m = helper.m
        self._pairwise = helper._pairwise
        self.selected = np.zeros(self.n, dtype=np.uint8)
        self.covered = np.zeros(self.m, dtype=bool)
        self.coverage_weight = 0.0
        self.distinguished = np.zeros(self._pairwise.pair_weights.shape[0], dtype=bool)
        self.distinguished_weight = 0.0

    def reset(self) -> None:
        self.selected.fill(0)
        self.covered.fill(False)
        self.coverage_weight = 0.0
        if self.distinguished.size:
            self.distinguished.fill(False)
        self.distinguished_weight = 0.0

    def build_from_mask(self, mask: np.ndarray) -> None:
        self.reset()
        sel = np.asarray(mask, dtype=np.uint8)
        if sel.shape != (self.n,):
            raise ValueError("mask must have shape (n,)")
        self.selected = sel.copy()
        if sel.any():
            cols = sel.astype(bool)
            self.covered = (self.helper.D_bool[:, cols].sum(axis=1) > 0)
            if self.covered.any():
                self.coverage_weight = float((self.helper.probs[self.covered]).sum())
            if self._pairwise.diff_matrix.size > 0:
                diff_counts = sel @ self._pairwise.diff_matrix
                self.distinguished = diff_counts > 0
                if self.distinguished.any():
                    self.distinguished_weight = float(self._pairwise.pair_weights[self.distinguished].sum())
        else:
            self.coverage_weight = 0.0
            if self.distinguished.size:
                self.distinguished.fill(False)
                self.distinguished_weight = 0.0

    def current_fdr(self) -> float:
        if self.helper.total_prob <= 0.0:
            return 0.0
        return self.coverage_weight / self.helper.total_prob

    def current_fir(self) -> float:
        if self._pairwise.total_weight <= 0.0:
            return 0.0
        return self.distinguished_weight / self._pairwise.total_weight

    def gain_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        uncovered_probs = np.where(~self.covered, self.helper.probs, 0.0)
        coverage_weight_gain = self.helper.D_bool.T @ uncovered_probs
        coverage_weight_gain[self.selected == 1] = 0.0
        if self.helper.total_prob > 0.0:
            fdr_gain = coverage_weight_gain / self.helper.total_prob
        else:
            fdr_gain = np.zeros(self.n, dtype=float)
        if self._pairwise.diff_matrix.size > 0 and self._pairwise.total_weight > 0.0:
            remaining = np.where(~self.distinguished, self._pairwise.pair_weights, 0.0)
            fir_weight_gain = self._pairwise.diff_matrix @ remaining
            fir_weight_gain[self.selected == 1] = 0.0
            fir_gain = fir_weight_gain / self._pairwise.total_weight
        else:
            fir_gain = np.zeros(self.n, dtype=float)
        return fdr_gain.astype(float), fir_gain.astype(float)

    def add(self, idx: int) -> None:
        if self.selected[idx] == 1:
            return
        self.selected[idx] = 1
        col = self.helper.D_bool[:, idx].astype(bool)
        new_cover = (~self.covered) & col
        if new_cover.any():
            self.coverage_weight += float((self.helper.probs[new_cover]).sum())
            self.covered[new_cover] = True
        if self._pairwise.diff_matrix.size > 0:
            diff = (self._pairwise.diff_matrix[idx] == 1) & (~self.distinguished)
            if diff.any():
                self.distinguished[diff] = True
                self.distinguished_weight += float(self._pairwise.pair_weights[diff].sum())

    def selected_mask(self) -> np.ndarray:
        return self.selected.copy()


def test_adjacency_matrix(D: np.ndarray) -> np.ndarray:
    """Construct an adjacency matrix linking tests with shared fault coverage."""

    D_bool = (np.asarray(D, dtype=np.uint8) > 0).astype(np.uint8)
    adj = D_bool.T @ D_bool
    adj = (adj > 0).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj
