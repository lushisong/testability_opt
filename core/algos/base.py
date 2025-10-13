# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Any
import time
import numpy as np
from core.metrics import fdr, fir, cost

@dataclass
class AlgoResult:
    name: str
    selected: np.ndarray  # 0/1 vector shape (n,)
    fdr: float
    fir: float
    cost: float
    runtime_sec: float
    extra: Dict[str, Any]

class BaseAlgo:
    name: str = "Base"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None, **kwargs) -> AlgoResult:
        raise NotImplementedError

    @staticmethod
    def _wrap_result(name, selected, D, probs, costs, t0, extra=None) -> AlgoResult:
        t1 = time.perf_counter()
        return AlgoResult(
            name=name,
            selected=selected.astype(int),
            fdr=fdr(selected, D, probs),
            fir=fir(selected, D, probs),
            cost=cost(selected, costs),
            runtime_sec=t1 - t0,
            extra=extra or {}
        )
