# -*- coding: utf-8 -*-
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.algos import nn_guided as guided_module
from core.algos.greedy import GreedyAlgo
from core.algos.nn_guided import NNGuidedAlgo
from core.algos.nn_guided_offline import NNGuidedOfflineAlgo
from experiments.train_offline import train_guided_offline


def _reset_guided_cache() -> None:
    guided_module._MODEL_CACHE.clear()
    guided_module._CACHE_STATS["queries"] = 0
    guided_module._CACHE_STATS["hits"] = 0


def _benchmark_matrix():
    D = np.array(
        [
            [0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    probs = np.array([0.10073853, 0.11507934, 0.22491837, 0.24117717, 0.13828545, 0.17980114])
    probs = probs / probs.sum()
    costs = np.array([1.44142846, 1.79079913, 0.66167229, 1.34251224, 1.71249411, 1.66134068])
    return D, probs, costs


def test_nn_guided_beats_greedy_under_budget():
    _reset_guided_cache()
    D, probs, costs = _benchmark_matrix()
    tau_d, tau_i, budget = 0.7, 0.6, 3.0

    res_g = GreedyAlgo().run(D, probs, costs, tau_d, tau_i, seed=0, budget=budget)
    assert res_g.fdr >= tau_d
    assert res_g.fir < tau_i - 1e-6

    res_nn = NNGuidedAlgo().run(D, probs, costs, tau_d, tau_i, seed=0, budget=budget)
    assert res_nn.fdr >= tau_d
    assert res_nn.fir >= tau_i
    assert res_nn.cost <= res_g.cost
    assert res_nn.selected.tolist() == [0, 0, 1, 1, 0, 0]


def test_nn_guided_cache_reuse_across_constraints():
    _reset_guided_cache()
    D, probs, costs = _benchmark_matrix()
    budget = 3.0

    first = NNGuidedAlgo().run(D, probs, costs, 0.7, 0.6, seed=42, budget=budget)
    assert first.extra["cached"] is False
    assert first.extra["train_samples"] > 0
    assert first.cache_hits == 0
    assert first.cache_queries == 1

    second = NNGuidedAlgo().run(D, probs, costs, 0.82, 0.65, seed=43, budget=budget)
    assert second.extra["cached"] is True
    assert second.extra["train_samples"] == 0
    assert second.extra["train_time_sec"] == 0.0
    assert second.cache_hits >= 1
    assert second.cache_queries == 2
    assert second.extra["profile_distance"] == pytest.approx(0.0)

    D_variant = D.copy()
    D_variant[0, 0] = 1
    third = NNGuidedAlgo().run(
        D_variant,
        probs,
        costs,
        0.72,
        0.62,
        seed=44,
        budget=budget,
        structure_tolerance=0.2,
    )
    assert third.extra["cached"] is True
    assert third.extra["profile_distance"] > 0.0
    assert third.extra["train_samples"] == 0
    assert third.cache_hits >= 2
    assert third.cache_queries == 3


def test_nn_guided_offline_pretraining_handles_variants(tmp_path):
    _reset_guided_cache()
    D, probs, costs = _benchmark_matrix()
    model_path = tmp_path / "nn_guided_family.npz"

    train_guided_offline(
        D,
        probs,
        costs,
        out_path=str(model_path),
        synth_samples=200,
        epochs=120,
        hidden=32,
        seed=0,
    )

    offline_algo = NNGuidedOfflineAlgo()
    base = offline_algo.run(
        D,
        probs,
        costs,
        0.7,
        0.6,
        seed=0,
        budget=3.0,
        model_path=str(model_path),
    )
    assert base.extra["fallback"] is False
    assert base.extra["profile_distance"] == pytest.approx(0.0)

    variant = D.copy()
    variant[0, 0] = 1
    reused = offline_algo.run(
        variant,
        probs,
        costs,
        0.72,
        0.62,
        seed=1,
        budget=3.0,
        model_path=str(model_path),
        structure_tolerance=0.2,
    )
    assert reused.extra["fallback"] is False
    assert reused.extra["profile_distance"] > 0.0
    assert reused.selected.tolist() == base.selected.tolist()
