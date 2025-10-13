# -*- coding: utf-8 -*-
import numpy as np

def per_test_features(D: np.ndarray, probs: np.ndarray, costs: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """
    为每个候选测试列构造与上下文相关的特征，用于 NN 打分或MILP Hint。
    返回 shape (n, fdim)
    """
    m, n = D.shape
    sel_cols = np.where(selected == 1)[0]
    covered = (D[:, sel_cols].sum(axis=1) > 0) if sel_cols.size > 0 else np.zeros(m, dtype=bool)

    feats = []
    if sel_cols.size == 0:
        sig = np.zeros(m, dtype=np.int64)
    else:
        sig = D[:, sel_cols].dot(1 << np.arange(sel_cols.size, dtype=np.int64))

    for j in range(n):
        col = D[:, j]
        new_cover = (~covered) & (col == 1)
        w_cov = float((probs[new_cover]).sum())
        cnt_cov = int(new_cover.sum())
        sep_gain = 0.0
        if sel_cols.size > 0:
            sig_new = D[:, np.r_[sel_cols, j]].dot(1 << np.arange(sel_cols.size + 1, dtype=np.int64))
            sep_gain = float(np.mean(sig != sig_new))
        else:
            sep_gain = float((probs[col == 1]).sum())
        feats.append([1.0, costs[j], w_cov, cnt_cov, sep_gain])
    return np.array(feats, dtype=float)

