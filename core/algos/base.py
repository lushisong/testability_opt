# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
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
    trajectory: Optional[List[Dict[str, float]]] = None
    primal_gap: Optional[float] = None
    dual_gap: Optional[float] = None
    cache_hits: Optional[int] = None
    cache_queries: Optional[int] = None

class BaseAlgo:
    name: str = "Base"

    def run(self, D, probs, costs, tau_d, tau_i, seed=None, **kwargs) -> AlgoResult:
        raise NotImplementedError

    @staticmethod
    def _wrap_result(name, selected, D, probs, costs, t0, extra=None) -> AlgoResult:
        t1 = time.perf_counter()
        extra_dict = dict(extra or {})
        trajectory = extra_dict.pop("trajectory", None)
        primal_gap = extra_dict.pop("primal_gap", None)
        dual_gap = extra_dict.pop("dual_gap", None)
        cache_hits = extra_dict.pop("cache_hits", extra_dict.pop("model_cache_hits", None))
        cache_queries = extra_dict.pop("cache_queries", extra_dict.pop("model_cache_queries", None))
        return AlgoResult(
            name=name,
            selected=selected.astype(int),
            fdr=fdr(selected, D, probs),
            fir=fir(selected, D, probs),
            cost=cost(selected, costs),
            runtime_sec=t1 - t0,
            extra=extra_dict,
            trajectory=trajectory,
            primal_gap=primal_gap,
            dual_gap=dual_gap,
            cache_hits=cache_hits,
            cache_queries=cache_queries
        )
