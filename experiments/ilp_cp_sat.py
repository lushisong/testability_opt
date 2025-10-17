# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from ortools.sat.python import cp_model

class AnytimeRecorder(cp_model.CpSolverSolutionCallback):
    def __init__(self, obj_sign: int = 1):
        super().__init__()
        self.t0 = time.perf_counter()
        self.traj: List[Tuple[float, float]] = []
        self.obj_sign = obj_sign

    def on_solution_callback(self):
        t = time.perf_counter() - self.t0
        val = self.obj_sign * self.objective_value()
        self.traj.append((t, val))

def solve_tp_mip_cp_sat(D: np.ndarray, probs: np.ndarray, costs: np.ndarray,
                        tau_d: float, tau_i: float,
                        time_limit_s: float = 10.0,
                        x_hint: Optional[np.ndarray] = None,
                        log: bool = False,
                        budget: Optional[float] = None,
                        num_workers: int = 8,
                        use_callback: bool = True) -> Dict[str, Any]:
    """
    MILP 等价CP-SAT建模:
      x_j ∈ {0,1} 选择测试
      y_i 表示检测到
      u_ik 区分对
      z_i 可隔离
    目标: 最小化成本 sum c_j x_j
    约束: FDR ≥ τ_d, FIR ≥ τ_i
    """
    m, n = D.shape
    model = cp_model.CpModel()

    # Variables
    x = [model.NewBoolVar(f"x_{j}") for j in range(n)]
    y = [model.NewBoolVar(f"y_{i}") for i in range(m)]
    z = [model.NewBoolVar(f"z_{i}") for i in range(m)]
    u = {}
    for i in range(m):
        for k in range(i+1, m):
            u[(i,k)] = model.NewBoolVar(f"u_{i}_{k}")

    # y_i ≤ sum_j d_ij x_j
    for i in range(m):
        # sum(d_ij x_j) >= y_i
        model.Add(sum(D[i, j] * x[j] for j in range(n)) >= y[i])

    # pairwise distinguishability: sum_j |d_ij - d_kj| x_j ≥ u_ik
    for i in range(m):
        for k in range(i+1, m):
            coef = [abs(int(D[i, j]) - int(D[k, j])) for j in range(n)]
            model.Add(sum(coef[j] * x[j] for j in range(n)) >= u[(i,k)])

    # z_i 与 u 关系: sum_{k≠i} u_ik ≥ (m-1) z_i  (L=1)
    for i in range(m):
        lhs = []
        for k in range(m):
            if k == i: 
                continue
            ii, kk = (i, k) if i < k else (k, i)
            lhs.append(u[(ii, kk)])
        model.Add(sum(lhs) >= (m-1) * z[i])

    # FDR ≥ τ_d
    W = float(probs.sum())
    scale = 1_000_000  # 线性化比例，避免浮点
    w_scale = [int(round(p * scale)) for p in probs]
    model.Add(sum(w_scale[i] * y[i] for i in range(m)) >= int(round(tau_d * W * scale)))

    # FIR ≥ τ_i (按 z_i 加权)
    model.Add(sum(w_scale[i] * z[i] for i in range(m)) >= int(round(tau_i * W * scale)))

    # Objective
    c_scale = 1_000
    c_int = [int(round(float(costs[j]) * c_scale)) for j in range(n)]
    # 预算约束（可选）
    if budget is not None:
        model.Add(sum(c_int[j] * x[j] for j in range(n)) <= int(round(float(budget) * c_scale)))
    model.Minimize(sum(c_int[j] * x[j] for j in range(n)))

    # Hints (Neural Diving)
    if x_hint is not None:
        for j in range(n):
            if x_hint[j] in (0, 1):
                model.AddHint(x[j], int(x_hint[j]))

    # Solve
    solver = cp_model.CpSolver()
    if time_limit_s is not None and time_limit_s > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(max(1, num_workers))
    solver.parameters.log_search_progress = log

    traj: List[Tuple[float, float]] = []
    if use_callback:
        cb = AnytimeRecorder(obj_sign=1)
        status = solver.SolveWithSolutionCallback(model, cb)
        traj = cb.traj
    else:
        t0 = time.perf_counter()
        status = solver.Solve(model)
        traj = [(time.perf_counter() - t0, solver.ObjectiveValue())]

    selected = np.array([int(solver.BooleanValue(x[j])) for j in range(n)], dtype=int)
    obj = sum(costs[j] * selected[j] for j in range(n))
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    return {
        "status": int(status),
        "selected": selected,
        "objective_cost": float(obj),
        "anytime_traj": traj,
        "feasible": feasible,
    }

