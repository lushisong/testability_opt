# -*- coding: utf-8 -*-
"""CP-SAT formulation for the testability optimization MIP."""

from typing import Dict, Any, Optional, Tuple, List

import time
import numpy as np
from ortools.sat.python import cp_model


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

    solver = cp_model.CpSolver()
    if time_limit_s is not None and time_limit_s > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = max(1, int(num_workers))
    solver.parameters.log_search_progress = log

    cb = AnytimeRecorder(obj_sign=1)
    solve_t0 = time.perf_counter()
    status = solver.SolveWithSolutionCallback(model, cb)
    solve_time = time.perf_counter() - solve_t0

    selected = np.array([int(solver.BooleanValue(x[j])) for j in range(n)], dtype=int)
    obj = sum(float(costs[j]) * selected[j] for j in range(n))
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    return {
        "status": int(status),
        "selected": selected,
        "objective_cost": float(obj),
        "anytime_traj": cb.traj,
        "feasible": feasible,
        "solve_time": float(solve_time),
    }

