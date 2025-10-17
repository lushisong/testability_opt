# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")

from core.benchmark import ALGO_REGISTRY
from core.metrics import fdr as compute_fdr, fir as compute_fir, cost as compute_cost


def _build_case1():
    rows = [
        "1111111111111111111011100",
        "1011111111111111110111000",
        "1001000000001111110001100",
        "1111111111111111110000011",
        "1011111111111111110011100",
        "0001111111111111110011111",
        "1001111111111111110000001",
        "0000011111111111110011100",
        "0000000001111111100000011",
        "0000000000111111100011100",
        "0000000000011111100000001",
        "0000000000001111100011101",
        "0000000000000111100000001",
        "0000000000000011000011011",
        "0000000010001111100011000",
    ]
    D = np.array([[int(ch) for ch in row] for row in rows], dtype=int)
    probs = np.array(
        [
            0.0010,
            0.0020,
            0.0010,
            0.0010,
            0.0030,
            0.0010,
            0.0010,
            0.0020,
            0.0010,
            0.0010,
            0.0010,
            0.0025,
            0.0010,
            0.0010,
            0.0020,
        ],
        dtype=float,
    )
    costs = np.ones(D.shape[1], dtype=float)
    return {
        "name": "case1",
        "D": D,
        "probs": probs,
        "costs": costs,
        "tau_d": 0.5,
        "tau_i": 0.2,
    }


def _build_case2():
    rows = [
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    ]
    D = np.array(rows, dtype=int)
    probs = np.full(D.shape[0], 0.1, dtype=float)
    costs = np.ones(D.shape[1], dtype=float)
    return {
        "name": "case2",
        "D": D,
        "probs": probs,
        "costs": costs,
        "tau_d": 0.4,
        "tau_i": 0.2,
    }


TEST_CASES = [_build_case1(), _build_case2()]
ALGO_TEST_KWARGS = {
    "Firefly": {"pop_size": 6, "max_iter": 10},
    "BinaryPSO": {"pop_size": 6, "max_iter": 10},
    "NN-Guided": {"synth_samples": 40, "epochs": 10, "hidden": 8},
    "NN-MIP": {"time_limit_s": 1.0, "epochs": 60, "hidden": 16, "hint_th": 0.5},
    "NN-Guided_Offline": {},
    "NN-MIP_Offline": {"time_limit_s": 1.0},
}


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("algo_name", sorted(ALGO_REGISTRY.keys()))
def test_algorithms_handle_defined_cases(case, algo_name):
    algo_proto = ALGO_REGISTRY[algo_name]
    algo = algo_proto.__class__()

    D = case["D"]
    probs = case["probs"]
    costs = case["costs"]
    tau_d = case["tau_d"]
    tau_i = case["tau_i"]

    extra_kwargs = ALGO_TEST_KWARGS.get(algo_name, {})
    # 若 NN-MIP 系列但未安装 ortools，则跳过
    if algo_name in ("NN-MIP", "NN-MIP_Offline"):
        try:
            import ortools.sat.python.cp_model  # noqa: F401
        except Exception:
            pytest.skip("ortools not installed; skip NN-MIP")
    result = algo.run(D, probs, costs, tau_d, tau_i, seed=123, **extra_kwargs)

    assert result.selected.shape == (D.shape[1],)
    assert set(np.unique(result.selected)).issubset({0, 1})
    assert np.isfinite(result.runtime_sec)
    assert np.isfinite(result.cost)

    # 验证度量与手工计算一致
    assert result.fdr == pytest.approx(compute_fdr(result.selected, D, probs))
    assert result.fir == pytest.approx(compute_fir(result.selected, D, probs))
    assert result.cost == pytest.approx(compute_cost(result.selected, costs))

    # 所有指标都应在合理范围内
    assert 0.0 <= result.fdr <= 1.0
    assert 0.0 <= result.fir <= 1.0
    assert result.cost >= 0.0
