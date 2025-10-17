# -*- coding: utf-8 -*-
"""Offline variant that disables model caching for reproducibility."""

from __future__ import annotations

from core.algos.neural_mip import NeuralMIPAlgo


class NNHintMIPOfflineAlgo(NeuralMIPAlgo):
    name = "NN-MIP-Offline"

    def run(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        kwargs.setdefault("use_cache", False)
        if "time_limit_s" in kwargs:
            kwargs["solver_time_limit"] = kwargs.pop("time_limit_s")
        if "hint_th" in kwargs:
            kwargs["hint_threshold"] = kwargs.pop("hint_th")
        return super().run(*args, **kwargs)
