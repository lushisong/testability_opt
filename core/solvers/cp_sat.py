# -*- coding: utf-8 -*-
"""CP-SAT formulation for the testability optimization MIP."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Sequence, Union

import time
import warnings
import numpy as np
from ortools.sat.python import cp_model


@dataclass
class BranchingStrategy:
    """Lightweight wrapper around CP-SAT decision strategies.

    Parameters
    ----------
    order : Sequence[int]
        Desired ordering for the Boolean decision variables. Elements refer to
        the index in the solver's ``x`` vector. Missing indices will be
        appended in their natural order.
    var_strategy : Union[int, str]
        Variable selection strategy from ``cp_model.DecisionStrategyProto``.
        Either an enum value or the corresponding string name, e.g.
        ``"CHOOSE_FIRST"``.
    domain_strategy : Union[int, str]
        Domain reduction strategy from ``cp_model.DecisionStrategyProto``.
    """

    order: Sequence[int]
    var_strategy: Union[int, str] = cp_model.CHOOSE_FIRST
    domain_strategy: Union[int, str] = cp_model.SELECT_MIN_VALUE

    def normalize(self, num_vars: int) -> Tuple[List[int], int, int]:
        seen = set()
        ordered: List[int] = []
        for idx in self.order:
            if 0 <= int(idx) < num_vars and int(idx) not in seen:
                ordered.append(int(idx))
                seen.add(int(idx))
        for idx in range(num_vars):
            if idx not in seen:
                ordered.append(idx)
        vs = self._resolve_strategy(self.var_strategy, cp_model.CHOOSE_FIRST)
        ds = self._resolve_strategy(self.domain_strategy, cp_model.SELECT_MIN_VALUE)
        return ordered, vs, ds

    @staticmethod
    def _resolve_strategy(value: Union[int, str], default: int) -> int:
        if isinstance(value, str):
            name = value.upper()
            if not name.startswith("SELECT") and not name.startswith("CHOOSE"):
                name = name.replace("-", "_")
            if not hasattr(cp_model, name):
                raise ValueError(f"Unknown decision strategy '{value}'")
            return int(getattr(cp_model, name))
        return int(value) if value is not None else int(default)


class AnytimeRecorder(cp_model.CpSolverSolutionCallback):
    """Records the anytime objective trajectory while CP-SAT searches."""

    def __init__(self, obj_sign: int = 1):
        super().__init__()
        self.t0 = time.perf_counter()
        self.traj: List[Tuple[float, float]] = []
        self.obj_sign = obj_sign

    def on_solution_callback(self):
        t = time.perf_counter() - self.t0
        val = self.obj_sign * self.ObjectiveValue()
        self.traj.append((t, val))


def solve_tp_mip_cp_sat(
    D: np.ndarray,
    probs: np.ndarray,
    costs: np.ndarray,
    tau_d: float,
    tau_i: float,
    time_limit_s: float = 10.0,
    x_hint: Optional[np.ndarray] = None,
    log: bool = False,
    num_workers: int = 8,
    branching: Optional[BranchingStrategy] = None,
) -> Dict[str, Any]:
    """Solve the test point selection MIP using Google OR-Tools CP-SAT.

    Parameters
    ----------
    D, probs, costs : ndarray
        Problem definition matrices/vectors.
    tau_d, tau_i : float
        Detection and isolation thresholds.
    time_limit_s : float, optional
        Wall-clock time limit in seconds.
    x_hint : ndarray, optional
        Optional {0,1} vector used as CP-SAT search hint (Neural Diving).
    log : bool, optional
        Whether to enable CP-SAT's verbose search log.
    num_workers : int, optional
        Number of parallel search workers.

    Returns
    -------
    dict
        Contains solver status, selected mask, objective cost, anytime trajectory
        and feasibility flag together with the wall-clock solve time.
    """

    m, n = D.shape
    model = cp_model.CpModel()

    # Variables
    x = [model.NewBoolVar(f"x_{j}") for j in range(n)]
    y = [model.NewBoolVar(f"y_{i}") for i in range(m)]
    z = [model.NewBoolVar(f"z_{i}") for i in range(m)]
    u = {}
    for i in range(m):
        for k in range(i + 1, m):
            u[(i, k)] = model.NewBoolVar(f"u_{i}_{k}")

    # Detection coverage constraints: y_i <= sum_j d_ij x_j
    for i in range(m):
        model.Add(sum(int(D[i, j]) * x[j] for j in range(n)) >= y[i])

    # Pairwise distinguishability constraints
    for i in range(m):
        for k in range(i + 1, m):
            coef = [abs(int(D[i, j]) - int(D[k, j])) for j in range(n)]
            model.Add(sum(coef[j] * x[j] for j in range(n)) >= u[(i, k)])

    # Isolation indicator linking
    for i in range(m):
        lhs = []
        for k in range(m):
            if k == i:
                continue
            ii, kk = (i, k) if i < k else (k, i)
            lhs.append(u[(ii, kk)])
        model.Add(sum(lhs) >= (m - 1) * z[i])

    # Detection threshold
    W = float(np.asarray(probs, dtype=float).sum())
    scale = 1_000_000
    w_scale = [int(round(float(p) * scale)) for p in probs]
    model.Add(sum(w_scale[i] * y[i] for i in range(m)) >= int(round(tau_d * W * scale)))

    # Isolation threshold
    model.Add(sum(w_scale[i] * z[i] for i in range(m)) >= int(round(tau_i * W * scale)))

    # Objective: minimise total cost
    c_scale = 1_000
    c_int = [int(round(float(costs[j]) * c_scale)) for j in range(n)]
    model.Minimize(sum(c_int[j] * x[j] for j in range(n)))

    # Hint integration
    if x_hint is not None:
        hint_arr = np.asarray(x_hint).reshape(-1)
        if hint_arr.size != n:
            raise ValueError("x_hint must have length equal to number of tests")
        for j in range(n):
            val = int(round(float(hint_arr[j])))
            if val in (0, 1):
                model.AddHint(x[j], val)

    if branching is not None:
        ordered, var_strategy, domain_strategy = branching.normalize(n)
        model.AddDecisionStrategy(
            [x[idx] for idx in ordered],
            var_strategy,
            domain_strategy,
        )

    solver = cp_model.CpSolver()
    if time_limit_s is not None and time_limit_s > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = max(1, int(num_workers))
    solver.parameters.log_search_progress = log

    cb = AnytimeRecorder(obj_sign=1)
    solve_t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="solve_with_solution_callback is deprecated",
            category=DeprecationWarning,
        )
        try:
            status = solver.Solve(model, cb)
        except TypeError:
            status = solver.SolveWithSolutionCallback(model, cb)
    solve_time = time.perf_counter() - solve_t0

    selected = np.array([int(solver.BooleanValue(x[j])) for j in range(n)], dtype=int)
    obj = sum(float(costs[j]) * selected[j] for j in range(n))
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    result = {
        "status": int(status),
        "selected": selected,
        "objective_cost": float(obj),
        "anytime_traj": cb.traj,
        "feasible": feasible,
        "solve_time": float(solve_time),
    }
    if branching is not None:
        result["branching_strategy"] = {
            "order": ordered,
            "var_strategy": int(var_strategy),
            "domain_strategy": int(domain_strategy),
        }
    return result
