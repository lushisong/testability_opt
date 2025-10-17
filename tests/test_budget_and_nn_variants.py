# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import os
import tempfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")

from core.data_io import random_dataset
from core.algos.greedy import GreedyAlgo
from core.algos.nn_guided import NNGuidedAlgo
from core.algos.nn_mip import NNHintMIPAlgo
from core.algos.nn_guided_offline import NNGuidedOfflineAlgo
from core.algos.nn_mip_offline import NNHintMIPOfflineAlgo
from experiments.train_offline import train_guided_offline, train_mip_offline


def test_greedy_respects_budget_zero():
    ds = random_dataset(8, 16, density=0.3, seed=7)
    algo = GreedyAlgo()
    res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.9, tau_i=0.8, seed=123, budget=0.0)
    assert res.cost == pytest.approx(0.0)


def test_nn_guided_cache_reuse():
    ds = random_dataset(10, 20, density=0.3, seed=11)
    algo = NNGuidedAlgo()
    r1 = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.8, tau_i=0.6, seed=1)
    r2 = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.85, tau_i=0.65, seed=2)
    assert r1.extra.get("cached", False) in (False, 0)
    assert r2.extra.get("cached", False) in (True, 1)


@pytest.mark.skipif("ortools.sat.python.cp_model" not in sys.modules and pytest.importorskip("importlib").util.find_spec("ortools") is None, reason="ortools not installed")
def test_nn_mip_ui_safe_params():
    ds = random_dataset(8, 18, density=0.3, seed=3)
    algo = NNHintMIPAlgo()
    res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.8, tau_i=0.6, seed=0,
                   time_limit_s=1.0, num_workers=1, use_callback=False)
    assert res.selected.shape == (ds.D.shape[1],)


def test_offline_guided_pipeline(tmp_path: Path):
    ds = random_dataset(10, 22, density=0.3, seed=5)
    save_path = tmp_path / "nn_guided_offline.npz"
    train_guided_offline(ds.D, ds.fault_probs, ds.test_costs, str(save_path), synth_samples=30, epochs=20, hidden=12, seed=0)
    algo = NNGuidedOfflineAlgo()
    res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.7, tau_i=0.5, seed=0, budget=None, model_path=str(save_path))
    assert res.selected.shape == (ds.D.shape[1],)
    assert res.extra.get("fallback", False) is False


@pytest.mark.skipif("ortools.sat.python.cp_model" not in sys.modules and pytest.importorskip("importlib").util.find_spec("ortools") is None, reason="ortools not installed")
def test_offline_mip_pipeline(tmp_path: Path):
    ds = random_dataset(8, 18, density=0.3, seed=9)
    save_path = tmp_path / "nn_mip_offline.npz"
    train_mip_offline(ds.D, ds.fault_probs, ds.test_costs, str(save_path), epochs=50, hidden=16, seed=0)
    algo = NNHintMIPOfflineAlgo()
    res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d=0.7, tau_i=0.5, seed=0, time_limit_s=0.3,
                   model_path=str(save_path), num_workers=1, use_callback=False)
    assert res.selected.shape == (ds.D.shape[1],)
    assert res.extra.get("fallback", False) is False
