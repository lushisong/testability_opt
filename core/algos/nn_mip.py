# -*- coding: utf-8 -*-
"""Backward compatibility alias for the NeuralMIP algorithm."""

from __future__ import annotations

from core.algos.neural_mip import NeuralMIPAlgo


class NNHintMIPAlgo(NeuralMIPAlgo):
    name = "NN-MIP"

    def run(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        if "time_limit_s" in kwargs:
            kwargs["solver_time_limit"] = kwargs.pop("time_limit_s")
        if "hint_th" in kwargs:
            kwargs["hint_threshold"] = kwargs.pop("hint_th")
        return super().run(*args, **kwargs)
