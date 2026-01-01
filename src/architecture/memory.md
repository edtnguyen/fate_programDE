Added Pyro pipeline/model stubs based on `src/architecture/pyro_pipeline_skeleton.md`, `src/architecture/pyro_context.md`, and `src/architecture/pyro_model.md`.
Implemented observed-data preprocessing (step 1) in `scripts/pyro_io.py`, including guide mapping, k-based filtering, and padded guide inputs.
Implemented guide-level decomposition (step 3) and linear predictor (step 4) helpers in `src/models/pyro_model.py` via `build_guide_effects` and `compute_linear_predictor`.
Implemented steps 5–7 in `src/models/pyro_model.py` inside `fate_model`: priors, random-walk gene effects, guide deviations, softmax mapping, and soft-label likelihood.
Implemented steps 8–10 in `src/models/pyro_model.py`: minibatched SVI via `fit_svi` (uses `pyro.plate` subsampling).
Implemented step 11 in `src/models/pyro_model.py`: `reconstruct_theta_samples` draws posterior theta samples and `export_gene_summary_for_ash` computes weighted summaries for a configurable contrast fate.
Added day-weight columns (`w0..w{D-1}`) to the gene summary CSV output for traceability.
Implemented step 12 in pipeline form: added `scripts/fit_pyro_export.py`, `scripts/run_ash.R`, and `scripts/rank_hits.py`, plus `config.yaml`, `Snakefile`, and conda env specs in `envs/`.
Implemented step 13 diagnostics in `scripts/run_diagnostics.py` (holdout cross-entropy for full vs nuisance-only model; optional guide permutation with a Python-only hit-rate estimate).
Added shared IO helpers in `scripts/pyro_io.py` and a `scripts/__init__.py` package marker.
Remaining stubs: `src/models/pyro_pipeline.py` only provides shared helpers (`make_k_centered`, `to_torch`).
