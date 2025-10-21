# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import time
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")
cp_model_mod = pytest.importorskip("ortools.sat.python.cp_model")

from core.data_io import random_dataset
from core.benchmark import create_algo


def _run_once(algo_name: str, ds, tau_d: float, tau_i: float, kwargs=None):
    algo = create_algo(algo_name)
    kwargs = kwargs or {}
    t0 = time.perf_counter()
    res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=2025, **kwargs)
    dt = time.perf_counter() - t0
    return res, dt


@pytest.mark.skipif("ortools.sat.python.cp_model" not in sys.modules and pytest.importorskip("importlib").util.find_spec("ortools") is None, reason="ortools not installed")
def test_nn_mip_runs_with_limit_and_feasible():
    m, n = 20, 40
    ds = random_dataset(m, n, density=0.30, seed=321)
    tau_d, tau_i = 0.90, 0.80
    # 限制很短时间，验证能稳定返回且时间受控
    res_n, tn = _run_once("NN-MIP", ds, tau_d, tau_i, {"time_limit_s": 0.2, "epochs": 30, "hidden": 8, "hint_th": 0.5, "num_workers": 1, "use_callback": False})
    assert res_n.selected.shape == (n,)
    assert tn <= 2.0  # 包含 Python 调度和启动开销


def test_nn_guided_caching_speeds_up():
    # 同一 D 下重复运行，第二次使用缓存应更快
    ds = random_dataset(40, 80, density=0.30, seed=111)
    tau_d, tau_i = 0.88, 0.80
    # 第一次使用更多样本以增加训练时间
    res1, t1 = _run_once("NN-Guided", ds, tau_d, tau_i, {"synth_samples": 160, "epochs": 30, "hidden": 12})
    res2, t2 = _run_once("NN-Guided", ds, tau_d, tau_i, {"synth_samples": 40, "epochs": 10, "hidden": 12})
    assert res2.extra.get("cached", False) in (True, 1)
    # 缓存后耗时应显著下降（留足冗余）
    assert t2 <= 0.75 * t1
