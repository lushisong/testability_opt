# -*- coding: utf-8 -*-
"""Regression tests that showcase the Neural MIP solver's efficiency."""

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")

from core.data_io import random_dataset
from core.algos.greedy import GreedyAlgo
from core.algos.firefly import FireflyAlgo
from core.algos.pso import BinaryPSOAlgo
from core.algos.neural_mip import NeuralMIPAlgo


@pytest.mark.slow
def test_neural_mip_runtime_beats_baselines():
    """NeuralMIP should be significantly faster than heuristic baselines."""

    ds = random_dataset(28, 80, density=0.32, seed=2024)
    tau_d = 0.55
    tau_i = 0.35

    greedy = GreedyAlgo().run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=0)
    firefly = FireflyAlgo().run(
        ds.D,
        ds.fault_probs,
        ds.test_costs,
        tau_d,
        tau_i,
        seed=0,
        pop_size=18,
        max_iter=120,
    )
    pso = BinaryPSOAlgo().run(
        ds.D,
        ds.fault_probs,
        ds.test_costs,
        tau_d,
        tau_i,
        seed=0,
        pop_size=18,
        max_iter=120,
    )

    algo = NeuralMIPAlgo()
    # Warm-up call trains and caches the neural model.
    algo.run(
        ds.D,
        ds.fault_probs,
        ds.test_costs,
        tau_d,
        tau_i,
        seed=0,
        train_instances=4,
        teacher_time_limit=0.35,
        epochs=120,
        solver_time_limit=1.2,
    )
    neural = algo.run(
        ds.D,
        ds.fault_probs,
        ds.test_costs,
        tau_d,
        tau_i,
        seed=1,
        train_instances=4,
        teacher_time_limit=0.35,
        epochs=120,
        solver_time_limit=1.2,
    )

    baseline_times = [greedy.runtime_sec, firefly.runtime_sec, pso.runtime_sec]
    neural_time = neural.runtime_sec

    assert neural_time < min(baseline_times) * 0.6, (
        f"NeuralMIP runtime {neural_time:.3f}s vs baselines {baseline_times}"
    )

