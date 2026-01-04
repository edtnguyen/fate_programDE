# Repository Guidelines

## Project Structure and Module Organization
- `src/` is the installable package; core modeling code lives in `src/models/`, data transforms in `src/features/`, plotting in `src/visualization/`, and architecture notes in `src/architecture/`.
- `scripts/` holds pipeline entrypoints (fit/export, shrinkage, ranking, diagnostics, simulations).
- `tests/` contains unit and integration tests.
- `Snakefile`, `config.yaml`, and `envs/` define the Snakemake workflow and conda environments.
- `data/` is for inputs; `out_fate_pipeline*` are generated outputs.

## Build, Test, and Development Commands
- `pip install -e .` installs the package locally so `src` imports resolve.
- `make requirements` installs Python dependencies; `make test_environment` checks the interpreter.
- `snakemake --use-conda --cores 1` runs the real-data pipeline; `snakemake --use-conda --cores 1 sim_all` runs the simulation chain.
- `make lint` runs flake8 over `src/`; `make clean` removes cached bytecode.

## Coding Style and Naming Conventions
- Python is 4-space indented; flake8 is the linter with max line length 79 and max complexity 10 (`tox.ini`).
- Use snake_case for modules, functions, and variables; CapWords for classes; UPPER_SNAKE for constants.
- Keep config keys in `config.yaml` aligned with pipeline scripts (for example, `sim_*`, `time_scale`).

## Testing Guidelines
- Tests use the standard library `unittest` and live in `tests/test_*.py`.
- Run a focused test with `python -m unittest tests/test_pyro_model_helpers.py`.
- Run the full suite with `PYTHON_BIN=python scripts/run_tests.sh` (includes `test_environment.py`).

## Commit and Pull Request Guidelines
- Recent commits use short, sentence-case subjects; optional scope prefixes like `model description:` appear in history. Follow that pattern.
- PRs should describe changes, list commands run, and call out config or output changes (especially under `out_fate_pipeline*`).

## Data, Config, and Security Notes
- Pipeline behavior is driven by `config.yaml`; keep paths local and avoid committing large raw data.
- If using S3 sync, set `BUCKET` and `PROFILE` in the `Makefile` or via environment overrides.
