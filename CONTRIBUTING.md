# Contributing to Lacuna

Lacuna is a solo-maintained research tool, but issues, bug reports, and pull
requests are welcome.

## Ways to contribute

- **Bug reports**: incorrect pocket geometry, crashes on valid input, install
  problems.
- **Benchmark results**: a new target, a new backend run, or a correction to
  an existing number. See "Honest reporting" below before opening one of these.
- **New backends or detectors**: the ensemble/detection/scoring stages are
  modular (`lacuna/ensemble/`, `lacuna/pockets/`); a new conformer source only
  needs to produce structures Lacuna can load.
- **Docs**: fixes to the README, [docs/BENCHMARKS.md](docs/BENCHMARKS.md), or
  docstrings.

## Development setup

```bash
git clone https://github.com/mooreneural/lacuna.git
cd lacuna
pip install -e ".[dev]"
```

Requires Python >=3.10. `dev` extras include `pytest`, `pytest-cov`, `ruff`,
and `mypy`. Optional backends (`openmm`, `boltz`) are separate extras; you
don't need them to run the core test suite.

## Running tests

```bash
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the full suite on Python 3.10, 3.11, and
3.12 for every push and pull request to `main`.

## Running benchmarks

Scripts live in `benchmarks/` (`cryptic_benchmark.py`,
`pocketminer_benchmark.py`, `cryptobench_benchmark.py`, and the fpocket
head-to-head scripts). Most auto-download their datasets to a gitignored data
directory on first run. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the
full current numbers and the commands that produced them.

## Code style

- `ruff` for linting (line length 100, target `py310`; config in
  `pyproject.toml`).
- Match the existing style in the file you're editing over imposing a new one.
- Keep comments to the non-obvious "why", not a restatement of the code.

## Honest reporting

This project's whole premise is that cryptic-pocket benchmarks are easy to
game (a metric-choice reversal is one of its own published findings). If your
PR changes a benchmark number:

- Report it under the size-robust criterion (`Jaccard >= 0.25` or centroid
  `<= 4A`), not just legacy recall. Both are computed by
  `benchmarks/metrics.py`.
- Prefer results with a target-level bootstrap CI (`paired_bootstrap_ci`) over
  a single run, especially for anything involving MD or other stochastic
  sampling.
- Negative and null results are welcome and get merged same as positive ones;
  several are already documented in `benchmarks/experiments/`.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make your change with tests where it makes sense.
3. Confirm `pytest tests/ -v` passes locally.
4. Open a PR describing what changed and why.

By submitting a pull request, you agree your contribution is licensed under
the project's AGPL-3.0-or-later license (see [LICENSE](LICENSE)).

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report
unacceptable behavior to clayton@tamu.edu.
