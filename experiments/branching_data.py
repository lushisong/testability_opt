# -*- coding: utf-8 -*-
"""Utilities to collect full strong branching style supervision data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from core.algos.utils import BinaryMetricHelper, test_adjacency_matrix
from core.data_io import Dataset, random_dataset


@dataclass
class BranchingSample:
    """Single branching state annotated with a target decision."""

    features: np.ndarray  # shape (n, fdim)
    adjacency: np.ndarray  # shape (n, n)
    selected_mask: np.ndarray  # shape (n,)
    candidate_mask: np.ndarray  # shape (n,)
    best_index: int

    def to_dict(self) -> dict:
        return {
            "features": self.features.tolist(),
            "adjacency": self.adjacency.tolist(),
            "selected_mask": self.selected_mask.astype(int).tolist(),
            "candidate_mask": self.candidate_mask.astype(int).tolist(),
            "best_index": int(self.best_index),
        }


class BranchingDataset:
    """Container for a list of :class:`BranchingSample` objects."""

    def __init__(self, samples: Iterable[BranchingSample] | None = None):
        self.samples: List[BranchingSample] = list(samples or [])

    def add(self, sample: BranchingSample) -> None:
        self.samples.append(sample)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        feats = np.array([s.features for s in self.samples], dtype=object)
        adjs = np.array([s.adjacency for s in self.samples], dtype=object)
        selected = np.array([s.selected_mask for s in self.samples], dtype=object)
        candidate = np.array([s.candidate_mask for s in self.samples], dtype=object)
        best = np.array([s.best_index for s in self.samples], dtype=np.int64)
        np.savez(path, features=feats, adjacency=adjs, selected=selected, candidate=candidate, best=best)

    @staticmethod
    def load_npz(path: str | Path) -> "BranchingDataset":
        data = np.load(path, allow_pickle=True)
        feats = data["features"].tolist()
        adjs = data["adjacency"].tolist()
        selected = data["selected"].tolist()
        candidate = data["candidate"].tolist()
        best = data["best"].tolist()
        samples = []
        for f, a, s, c, b in zip(feats, adjs, selected, candidate, best):
            samples.append(
                BranchingSample(
                    features=np.asarray(f, dtype=float),
                    adjacency=np.asarray(a, dtype=float),
                    selected_mask=np.asarray(s, dtype=int),
                    candidate_mask=np.asarray(c, dtype=int),
                    best_index=int(b),
                )
            )
        return BranchingDataset(samples)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = [s.to_dict() for s in self.samples]
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)


def _strong_branching_score(helper: BinaryMetricHelper, tracker_mask: np.ndarray, candidate_idx: Sequence[int]) -> np.ndarray:
    tracker = helper.new_tracker()
    tracker.build_from_mask(tracker_mask)
    fdr_gain, fir_gain = tracker.gain_vectors()
    scores = 0.5 * fdr_gain + 0.5 * fir_gain
    scores = np.asarray(scores, dtype=float)
    mask = np.full(helper.n, -np.inf, dtype=float)
    mask[np.asarray(candidate_idx, dtype=int)] = scores[np.asarray(candidate_idx, dtype=int)]
    return mask


def collect_branching_samples(ds: Dataset, rng: np.random.Generator, per_instance: int = 4) -> List[BranchingSample]:
    helper = BinaryMetricHelper(ds.D, ds.fault_probs, ds.test_costs)
    adjacency = test_adjacency_matrix(ds.D)
    n = helper.n
    samples: List[BranchingSample] = []
    for _ in range(per_instance):
        tracker = helper.new_tracker()
        selected = np.zeros(n, dtype=np.uint8)
        depth = rng.integers(low=0, high=max(1, n))
        available = list(range(n))
        rng.shuffle(available)
        for step in range(min(depth, n)):
            idx = available[step]
            selected[idx] = 1
            tracker.add(idx)
        while True:
            feats = helper.feature_matrix(selected)
            candidate_mask = (selected == 0).astype(np.uint8)
            candidates = np.where(candidate_mask == 1)[0]
            if candidates.size == 0:
                break
            scores = _strong_branching_score(helper, selected, candidates)
            best = int(np.argmax(scores))
            samples.append(
                BranchingSample(
                    features=feats.astype(float),
                    adjacency=adjacency.astype(float),
                    selected_mask=selected.astype(np.uint8),
                    candidate_mask=candidate_mask,
                    best_index=best,
                )
            )
            selected[best] = 1
            tracker.add(best)
            if tracker.current_fdr() >= 1.0 and tracker.current_fir() >= 1.0:
                break
    return samples


def collect_dataset(num_instances: int, m: int, n: int, density: float, seed: int, per_instance: int) -> BranchingDataset:
    rng = np.random.default_rng(seed)
    dataset = BranchingDataset()
    for k in range(num_instances):
        ds = random_dataset(m=m, n=n, density=density, seed=seed + k)
        samples = collect_branching_samples(ds, rng, per_instance=per_instance)
        for s in samples:
            dataset.add(s)
    return dataset


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Collect branching supervision data using synthetic instances.")
    parser.add_argument("--instances", type=int, default=32, help="Number of random datasets to sample")
    parser.add_argument("--m", type=int, default=20, help="Number of faults per dataset")
    parser.add_argument("--n", type=int, default=40, help="Number of tests per dataset")
    parser.add_argument("--density", type=float, default=0.3, help="Density of the detection matrix")
    parser.add_argument("--per_instance", type=int, default=6, help="Branching states generated per dataset")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out", type=str, default="branching_data.npz", help="Output .npz file")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON dump for inspection")
    args = parser.parse_args()

    dataset = collect_dataset(args.instances, args.m, args.n, args.density, args.seed, args.per_instance)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_npz(args.out)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(args.json)
    print(f"Saved {len(dataset.samples)} branching samples to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
