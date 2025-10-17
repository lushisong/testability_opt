# -*- coding: utf-8 -*-
"""Offline variant of the NN-guided heuristic reusing cached models."""

from __future__ import annotations

from core.algos.nn_guided import NNGuidedAlgo


class NNGuidedOfflineAlgo(NNGuidedAlgo):
    name = "NN-Guided-Offline"

    def run(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        kwargs.setdefault("use_cache", True)
        return super().run(*args, **kwargs)
