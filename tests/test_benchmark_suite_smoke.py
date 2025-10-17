# -*- coding: utf-8 -*-
from pathlib import Path
import json

import pytest


np = pytest.importorskip("numpy")

from experiments.benchmark_suite import run_suite


def test_benchmark_suite_smoke(tmp_path):
    config = {
        "name": "smoke",
        "seed": 4,
        "families": [
            {
                "name": "toy",
                "m": 6,
                "n": 5,
                "base_density": 0.4,
                "type": "local_perturbation",
                "perturbations": {"count": 2, "magnitude": 0.3, "radius": 0.5},
                "probabilities": {"low": 0.2, "high": 1.0},
                "costs": {"low": 1.0, "high": 3.0},
            }
        ],
        "tau_grid": [{"tau_d": 0.3, "tau_i": 0.1}],
        "budgets": [None, 2.5],
        "algorithms": ["Greedy", "Firefly"],
        "repeats": 1,
        "scheduler": "sequential",
    }

    result = run_suite(config, tmp_path)

    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["families"], "manifest should list generated families"
    first_family = manifest["families"][0]
    assert first_family["variants"], "family variants should not be empty"

    # check archive contains per-combo results
    archive_files = sorted(run_dir.glob("archive/**/budget_*.json"))
    assert archive_files, "expected archived result files"
    sample = json.loads(archive_files[0].read_text(encoding="utf-8"))
    assert sample and {"algo", "trajectory", "primal_gap", "dual_gap"}.issubset(sample[0])

    summary = result["summary"]
    csv_path = Path(summary["csv"])
    assert csv_path.exists()
    report_path = Path(summary["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "overall" in report and report["overall"], "summary report must contain per-algo stats"

    anytime_path = Path(summary["anytime"])
    anytime = json.loads(anytime_path.read_text(encoding="utf-8"))
    assert any(series for series in anytime.values()), "anytime trajectories should not be empty"

    # Figures should be generated for later manual inspection
    for fig_name in summary["figs"]:
        assert (csv_path.parent / fig_name).exists()
