# Repository Guidelines

## Project Structure & Module Organization
- `app.py` boots the PyQt5 desktop; it creates `data/` and `results/` for runtime artefacts.
- `core/` hosts algorithms, metrics, and IO helpers; add new solvers under `core/algos/` and register them in `core/benchmark.py`.
- `experiments/` contains CLI pipelines for matrix generation, benchmark runs, and training; each script exposes `--help`.
- UI code lives in `widgets/` (Python) and `ui/` (Qt `.ui` files); keep widget logic side-effect free so `tests/test_ui_tabs.py` can import it.
- Scenario-driven tests sit in `tests/`; seed data for notebooks or quick demos under `data/`, and store benchmark outputs in `results/`.

## Build, Test, and Development Commands
- `python app.py` launches the GUI for manual workflows.
- `python -m pytest` executes the suite; use `python -m pytest -m "not slow"` to skip heavy runs.
- `python -m experiments.benchmark_suite --config configs/toy.json --output results/toy_run` (replace paths as needed) generates a full benchmark archive.
- `python -m experiments.train_offline --config configs/offline.json --checkpoint results/offline/model.pt` trains offline hint models; reuse `results/` to keep artefacts out of git.
- `python -m experiments.branching_data --help` documents branching data preparation flags.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and type hints on public functions; prefer dataclasses for config blobs as seen in `experiments/benchmark_suite.py`.
- Name solver classes in PascalCase and suffix Qt classes with `Widget`; keep module names snake_case.
- Update `core/benchmark.py:ALGO_REGISTRY` and `widgets/algos_widget.py` when introducing new algorithms so they appear in both CLI and UI flows.

## Testing Guidelines
- New tests belong in `tests/` with `test_*.py` names; mirror the feature placement.
- Mark regressions that exceed a minute with `@pytest.mark.slow` to respect the existing marker in `pytest.ini`.
- Use pytest fixtures such as `tmp_path` for generated files; assert manifests and reports rather than comparing binary artefacts.

## Commit & Pull Request Guidelines
- Commits stay short, imperative, and scoped (e.g., `Enhance benchmarking suite`, `fix: stabilize offline trainer`); include a brief body if behavior changes.
- PRs must describe motivation, outline verification (`python -m pytest`, screenshots of updated tabs), and link relevant experiment configs or issues.
- Call out schema changes that alter saved results so reviewers can regenerate baselines.
