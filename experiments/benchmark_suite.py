# -*- coding: utf-8 -*-
"""Benchmark orchestration utilities.

This module provides a configurable pipeline for generating families of
coverage matrices ("D" matrices), running the benchmark suite under
multiple constraint settings, and archiving the resulting artefacts in a
structured, reproducible format.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np

from core.benchmark import ALGO_REGISTRY, run_benchmark, summarize_and_plot


@dataclass
class MatrixVariant:
    """Container for a generated coverage matrix variant."""

    name: str
    matrix: np.ndarray
    probs: np.ndarray
    costs: np.ndarray
    metadata: Dict[str, Any]


def load_config(config: str | os.PathLike | Dict[str, Any]) -> Dict[str, Any]:
    """Load a benchmark suite configuration from JSON or YAML."""

    if isinstance(config, dict):
        return config

    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "PyYAML is required to parse YAML configs. Install pyyaml or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping/dict")
    return data


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_probs(cfg: Dict[str, Any], m: int, rng: np.random.Generator) -> np.ndarray:
    spec = cfg.get("probabilities", {}) or {}
    if "values" in spec:
        arr = np.asarray(spec["values"], dtype=float)
        if arr.shape[0] != m:
            raise ValueError("probabilities.values length mismatch")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("probabilities.values must sum to > 0")
        return arr / total
    low = float(spec.get("low", 0.01))
    high = float(spec.get("high", 1.0))
    raw = rng.uniform(low, high, size=m)
    return raw / raw.sum()


def _generate_costs(cfg: Dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    spec = cfg.get("costs", {}) or {}
    if "values" in spec:
        arr = np.asarray(spec["values"], dtype=float)
        if arr.shape[0] != n:
            raise ValueError("costs.values length mismatch")
        return arr
    low = float(spec.get("low", 1.0))
    high = float(spec.get("high", 10.0))
    return rng.uniform(low, high, size=n)


def _generate_base_matrix(cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    m = int(cfg.get("m") or cfg.get("rows") or 32)
    n = int(cfg.get("n") or cfg.get("cols") or 32)
    density = float(cfg.get("base_density", cfg.get("density", 0.2)))
    base = rng.binomial(1, density, size=(m, n)).astype(int)
    ensure_cover = cfg.get("ensure_cover", True)
    if ensure_cover:
        for i in range(m):
            if base[i].sum() == 0:
                j = int(rng.integers(0, n))
                base[i, j] = 1
    return base


def _local_perturbations(
    base: np.ndarray,
    count: int,
    magnitude: float,
    radius: float,
    rng: np.random.Generator,
) -> Iterable[Tuple[np.ndarray, Dict[str, Any]]]:
    m, n = base.shape
    rows = max(1, int(m * radius))
    cols = max(1, int(n * radius))
    for idx in range(count):
        variant = base.copy()
        sel_rows = rng.choice(m, size=rows, replace=False)
        sel_cols = rng.choice(n, size=cols, replace=False)
        mask = np.zeros_like(variant, dtype=bool)
        mask[np.ix_(sel_rows, sel_cols)] = True
        flip_mask = mask & (rng.random(size=variant.shape) < magnitude)
        variant[flip_mask] = 1 - variant[flip_mask]
        yield variant, {
            "variant_type": "local_perturbation",
            "rows": rows,
            "cols": cols,
            "magnitude": magnitude,
            "index": idx,
        }


def generate_matrix_family(
    family_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[MatrixVariant]:
    """Create a set of related D matrices as specified by ``family_cfg``."""

    name = family_cfg.get("name", "family")
    base = _generate_base_matrix(family_cfg, rng)
    probs = _generate_probs(family_cfg, base.shape[0], rng)
    costs = _generate_costs(family_cfg, base.shape[1], rng)

    variants: List[MatrixVariant] = []
    include_base = family_cfg.get("include_base", True)
    if include_base:
        variants.append(
            MatrixVariant(
                name=f"{name}_base",
                matrix=base.copy(),
                probs=probs.copy(),
                costs=costs.copy(),
                metadata={"variant_type": "base"},
            )
        )

    family_type = family_cfg.get("type", "fixed")
    if family_type == "fixed":
        extra_count = max(0, int(family_cfg.get("count", 0)))
        for idx in range(extra_count):
            variants.append(
                MatrixVariant(
                    name=f"{name}_fixed_{idx}",
                    matrix=base.copy(),
                    probs=probs.copy(),
                    costs=costs.copy(),
                    metadata={"variant_type": "fixed", "index": idx},
                )
            )
    elif family_type in {"local", "local_perturbation"}:
        pert_cfg = family_cfg.get("perturbations", {})
        count = int(pert_cfg.get("count", family_cfg.get("count", 4)))
        magnitude = float(pert_cfg.get("magnitude", 0.05))
        radius = float(pert_cfg.get("radius", pert_cfg.get("fraction", 0.2)))
        for variant, meta in _local_perturbations(base, count, magnitude, radius, rng):
            variants.append(
                MatrixVariant(
                    name=f"{name}_perturbed_{meta['index']}",
                    matrix=variant,
                    probs=probs.copy(),
                    costs=costs.copy(),
                    metadata=meta,
                )
            )
    else:
        raise ValueError(f"Unsupported family type: {family_type}")

    return variants


def _budget_labels(budget: float | None) -> str:
    return "none" if budget is None else f"{budget:.3f}".rstrip("0").rstrip(".")


def run_suite(
    config: Dict[str, Any],
    output_dir: str | os.PathLike,
    config_path: str | os.PathLike | None = None,
) -> Dict[str, Any]:
    """Run the benchmark suite as described by ``config`` and archive results."""

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    root = _ensure_dir(Path(output_dir))
    run_name = config.get("name", "benchmark")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _ensure_dir(root / f"{run_name}_{timestamp}")
    configs_dir = _ensure_dir(run_dir / "configs")
    archive_dir = _ensure_dir(run_dir / "archive")
    summary_dir = _ensure_dir(run_dir / "summary")

    config_copy_path = configs_dir / "config.json"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    if config_path:
        src = Path(config_path)
        if src.exists():
            (configs_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    families = config.get("families", [])
    if not families:
        raise ValueError("Configuration must contain at least one family")

    tau_grid = config.get("tau_grid") or config.get("targets")
    if not tau_grid:
        raise ValueError("Configuration must define tau_grid/targets entries")
    budgets = config.get("budgets", [None])
    algos = config.get("algorithms") or list(ALGO_REGISTRY.keys())
    if not algos:
        raise ValueError("No algorithms specified in configuration")

    repeats = int(config.get("repeats", 3))
    base_seed = int(config.get("base_seed", config.get("seed", 0)))
    scheduler_cfg = config.get("scheduler", {})
    if isinstance(scheduler_cfg, str):
        scheduler_mode = scheduler_cfg
        scheduler_workers = None
    else:
        scheduler_mode = scheduler_cfg.get("mode", "sequential")
        scheduler_workers = scheduler_cfg.get("max_workers")

    manifest: Dict[str, Any] = {
        "run_name": run_name,
        "timestamp": timestamp,
        "seed": seed,
        "families": [],
    }
    all_results: List[Dict[str, Any]] = []

    for family_cfg in families:
        family_name = family_cfg.get("name", f"family_{len(manifest['families'])}")
        variants = generate_matrix_family(family_cfg, rng)
        family_entry = {"name": family_name, "variants": []}
        for variant in variants:
            variant_dir = _ensure_dir(archive_dir / family_name / variant.name)
            np.save(variant_dir / "D.npy", variant.matrix)
            np.save(variant_dir / "probs.npy", variant.probs)
            np.save(variant_dir / "costs.npy", variant.costs)
            (variant_dir / "metadata.json").write_text(
                json.dumps(variant.metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            variant_summary = {"name": variant.name, "metadata": variant.metadata, "runs": []}

            for tau in tau_grid:
                tau_d = float(tau.get("tau_d") or tau.get("fdr"))
                tau_i = float(tau.get("tau_i") or tau.get("fir"))
                for budget in budgets:
                    run_results = run_benchmark(
                        variant.matrix,
                        variant.probs,
                        variant.costs,
                        tau_d,
                        tau_i,
                        algos=algos,
                        repeats=repeats,
                        base_seed=base_seed,
                        budget=budget,
                        scheduler=scheduler_mode,
                        max_workers=scheduler_workers,
                    )
                    for row in run_results:
                        row.update(
                            {
                                "family": family_name,
                                "variant": variant.name,
                                "tau_d": tau_d,
                                "tau_i": tau_i,
                                "budget": row.get("budget", budget),
                            }
                        )
                    combo_dir = _ensure_dir(
                        variant_dir
                        / f"tau_d_{tau_d:.3f}".rstrip("0").rstrip(".")
                        / f"tau_i_{tau_i:.3f}".rstrip("0").rstrip(".")
                    )
                    combo_path = combo_dir / f"budget_{_budget_labels(budget)}.json"
                    combo_path.write_text(json.dumps(run_results, ensure_ascii=False, indent=2), encoding="utf-8")
                    variant_summary["runs"].append(
                        {
                            "tau_d": tau_d,
                            "tau_i": tau_i,
                            "budget": budget,
                            "results_path": str(combo_path.relative_to(run_dir)),
                        }
                    )
                    all_results.extend(run_results)
            family_entry["variants"].append(variant_summary)
        manifest["families"].append(family_entry)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_info = summarize_and_plot(all_results, str(summary_dir))
    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "summary": summary_info,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the extended benchmark suite")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML configuration file")
    parser.add_argument("--output", required=True, help="Directory to store artefacts")
    parser.add_argument(
        "--scheduler",
        choices=["sequential", "thread", "process"],
        help="Override scheduler mode defined in the config",
    )
    parser.add_argument("--max-workers", type=int, help="Override max_workers for threaded/process execution")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    if args.scheduler:
        cfg["scheduler"] = {"mode": args.scheduler, "max_workers": args.max_workers}
    elif args.max_workers is not None:
        sched = cfg.setdefault("scheduler", {})
        if isinstance(sched, dict):
            sched["max_workers"] = args.max_workers
        else:
            cfg["scheduler"] = {"mode": sched, "max_workers": args.max_workers}

    result = run_suite(cfg, args.output, config_path=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
