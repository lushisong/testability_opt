# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class Dataset:
    D: np.ndarray            # shape (m, n), entries 0/1
    fault_probs: np.ndarray  # shape (m,)
    test_costs: np.ndarray   # shape (n,)
    fault_names: List[str]
    test_names: List[str]

    def validate(self) -> None:
        m, n = self.D.shape
        assert self.fault_probs.shape == (m,)
        assert self.test_costs.shape == (n,)
        assert len(self.fault_names) == m
        assert len(self.test_names) == n
        assert ((self.D == 0) | (self.D == 1)).all(), "D must be 0/1"
        s = self.fault_probs.sum()
        if s <= 0:
            raise ValueError("Sum of fault probabilities must be > 0")
        # normalize for safety
        self.fault_probs = self.fault_probs / s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faults": [{"name": self.fault_names[i], "prob": float(self.fault_probs[i])}
                       for i in range(self.D.shape[0])],
            "tests": [{"name": self.test_names[j], "cost": float(self.test_costs[j])}
                      for j in range(self.D.shape[1])],
            "D": self.D.astype(int).tolist(),
        }

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "Dataset":
        faults = obj["faults"]
        tests = obj["tests"]
        D = np.array(obj["D"], dtype=np.int64)
        fault_names = [f["name"] for f in faults]
        test_names = [t["name"] for t in tests]
        probs = np.array([f["prob"] for f in faults], dtype=float)
        costs = np.array([t["cost"] for t in tests], dtype=float)
        ds = Dataset(D=D, fault_probs=probs, test_costs=costs,
                     fault_names=fault_names, test_names=test_names)
        ds.validate()
        return ds

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return Dataset.from_dict(obj)

def random_dataset(m: int, n: int, density: float = 0.3,
                   prob_scale: float = 1.0,
                   cost_low: float = 1.0, cost_high: float = 5.0,
                   seed: Optional[int] = None) -> Dataset:
    rng = np.random.default_rng(seed)
    D = (rng.random((m, n)) < density).astype(np.int64)
    # 保证每个故障至少可被一个测试覆盖
    for i in range(m):
        if D[i].sum() == 0:
            j = rng.integers(0, n)
            D[i, j] = 1
    probs_raw = rng.random(m)
    probs = probs_raw / probs_raw.sum()
    costs = rng.uniform(cost_low, cost_high, size=n)
    fault_names = [f"F{i+1}" for i in range(m)]
    test_names = [f"T{j+1}" for j in range(n)]
    ds = Dataset(D=D, fault_probs=probs, test_costs=costs,
                 fault_names=fault_names, test_names=test_names)
    ds.validate()
    return ds
