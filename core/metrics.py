# -*- coding: utf-8 -*-
from typing import Iterable, Tuple, Optional
import numpy as np

def fdr(selected: np.ndarray, D: np.ndarray, probs: np.ndarray) -> float:
    # selected: shape (n,), 0/1
    covered = (D[:, selected == 1].sum(axis=1) > 0).astype(np.int64)
    return float((probs * covered).sum() / probs.sum())

def fir(selected: np.ndarray, D: np.ndarray, probs: np.ndarray) -> float:
    # pairwise distinguishability
    m, n = D.shape
    sel_cols = np.where(selected == 1)[0]
    if sel_cols.size == 0:
        return 0.0
    rows = D[:, sel_cols]  # (m, k)
    # Use bitwise difference count > 0 to distinguish
    # Compute pairwise equality of signatures
    # For efficiency, hash signatures:
    sig = rows.dot(1 << np.arange(rows.shape[1], dtype=np.int64))
    # Pairs distinguishable if signatures differ
    # Weighted by probs
    total = 0.0
    good = 0.0
    for i in range(m):
        for j in range(i+1, m):
            w = float(probs[i] * probs[j])
            total += w
            if sig[i] != sig[j]:
                good += w
    if total == 0:
        return 0.0
    return good / total

def cost(selected: np.ndarray, costs: np.ndarray) -> float:
    return float((selected * costs).sum())

def redundancy(selected: np.ndarray, D: np.ndarray, rmin: int = 1) -> float:
    # 平均冗余度超出 rmin 的量
    covered_counts = D[:, selected == 1].sum(axis=1)
    extra = np.maximum(0, covered_counts - rmin)
    return float(np.mean(extra))

def marginal_gain(selected: np.ndarray, j: int, D: np.ndarray, probs: np.ndarray,
                  w_fdr: float = 0.5, w_fir: float = 0.5) -> Tuple[float, float, float]:
    if selected[j] == 1:
        return 0.0, 0.0, 0.0
    f0 = fdr(selected, D, probs)
    r0 = fir(selected, D, probs)
    s2 = selected.copy()
    s2[j] = 1
    f1 = fdr(s2, D, probs)
    r1 = fir(s2, D, probs)
    return (f1 - f0), (r1 - r0), (w_fdr * (f1 - f0) + w_fir * (r1 - r0))

def feasibility_gaps(selected: np.ndarray, D: np.ndarray, probs: np.ndarray,
                     tau_d: float, tau_i: float) -> Tuple[float, float]:
    fd = fdr(selected, D, probs)
    fr = fir(selected, D, probs)
    gd = max(0.0, tau_d - fd)
    gi = max(0.0, tau_i - fr)
    return gd, gi

def penalized_objective(selected: np.ndarray, D: np.ndarray, probs: np.ndarray,
                        costs: np.ndarray, tau_d: float, tau_i: float,
                        penalty: float = 1000.0) -> float:
    c = cost(selected, costs)
    gd, gi = feasibility_gaps(selected, D, probs, tau_d, tau_i)
    return c + penalty * (gd + gi)
