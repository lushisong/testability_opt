# -*- coding: utf-8 -*-
"""
离线训练工具：针对给定的 D/probs/costs 训练并保存 TinyMLP 模型，
供 NN-Guided_Offline 和 NN-MIP_Offline 使用。
"""
from __future__ import annotations

import os
import numpy as np
from typing import Tuple

from core.algos.utils import BinaryMetricHelper
from experiments.features import per_test_features
from experiments.models import TinyMLP, save_tinymlp


def train_guided_offline(D: np.ndarray, probs: np.ndarray, costs: np.ndarray,
                         out_path: str, synth_samples: int = 200, epochs: int = 300,
                         hidden: int = 32, seed: int = 0) -> Tuple[TinyMLP, np.ndarray, np.ndarray]:
    helper = BinaryMetricHelper(D, probs, costs)
    n = helper.n
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    tracker = helper.new_tracker()
    for _ in range(synth_samples):
        mask = (rng.random(n) < 0.4).astype(np.uint8)
        tracker.build_from_mask(mask)
        feats = helper.feature_matrix(tracker.selected)
        fdr_gain, fir_gain = tracker.gain_vectors()
        gains = 0.5 * fdr_gain + 0.5 * fir_gain
        avail = tracker.selected == 0
        if not np.any(avail):
            continue
        denom = np.maximum(helper.costs[avail], 1e-12)
        X_list.append(feats[avail])
        y_list.append(gains[avail] / denom)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    net = TinyMLP(in_dim=X.shape[1], hidden=hidden, lr=1e-2, seed=seed)
    net.fit_mse((X - mu) / sd, y, epochs=epochs, batch=128)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_tinymlp(out_path, net, mu, sd)
    return net, mu, sd


def train_mip_offline(D: np.ndarray, probs: np.ndarray, costs: np.ndarray,
                      out_path: str, epochs: int = 200, hidden: int = 32,
                      seed: int = 0) -> Tuple[TinyMLP, np.ndarray, np.ndarray]:
    n = D.shape[1]
    feats = per_test_features(D, probs, costs, selected=np.zeros(n, dtype=int))
    num = feats[:, 2] + feats[:, 4]
    den = feats[:, 1] + 1e-6
    y = num / den
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-8
    net = TinyMLP(in_dim=feats.shape[1], hidden=hidden, lr=1e-2, seed=seed)
    net.fit_mse((feats - mu) / sd, y, epochs=epochs, batch=128)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_tinymlp(out_path, net, mu, sd)
    return net, mu, sd

