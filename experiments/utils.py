# -*- coding: utf-8 -*-
import os, json, time
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tabulate_mean_std(rows: List[Dict[str, Any]], key: str, group_key: str = "algo") -> List[Tuple[str, float, float]]:
    algos = sorted({r[group_key] for r in rows})
    out = []
    for a in algos:
        vals = [r[key] for r in rows if r[group_key] == a]
        out.append((a, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0))
    return out

def performance_profile_relative_cost(rows: List[Dict[str, Any]], out_png: str):
    algos = sorted({r["algo"] for r in rows})
    reps = sorted({r["rep"] for r in rows})
    rel = {a: [] for a in algos}
    for rep in reps:
        sub = [r for r in rows if r["rep"] == rep]
        best = min(r["cost"] for r in sub)
        for a in algos:
            vals = [r["cost"] for r in sub if r["algo"] == a]
            rel[a].extend([v / (best + 1e-12) for v in vals])
    plt.figure()
    for a in algos:
        s = np.sort(np.array(rel[a], dtype=float))
        y = np.linspace(0, 1, len(s), endpoint=True)
        plt.plot(s, y, label=a)
    plt.xlabel("Relative Cost to Best (per repeat)")
    plt.ylabel("CDF")
    plt.title("Performance Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def anytime_plot(traj: Dict[str, List[Tuple[float, float]]], out_png: str, xlabel: str = "Time (s)", ylabel: str = "Objective"):
    plt.figure()
    for name, pairs in traj.items():
        if not pairs: 
            continue
        t = [p[0] for p in pairs]
        v = [p[1] for p in pairs]
        plt.plot(t, v, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Anytime Objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

