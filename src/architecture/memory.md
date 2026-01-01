Added Pyro pipeline/model stubs based on `src/architecture/pyro_pipeline_skeleton.md`, `src/architecture/pyro_context.md`, and `src/architecture/pyro_model.md`.
Implemented observed-data preprocessing (step 1) in `src/models/pyro_pipeline.py`: `build_padded_guides`, `filter_cells_by_k`, `make_k_centered`, and `to_torch` now follow the skeleton with basic validation.
Implemented latent indexing metadata (step 2) in `src/models/pyro_model.py` via `LatentIndex` plus fate name constants, exposing parameter shapes and fate index mappings.
Implemented guide-level decomposition (step 3) and linear predictor (step 4) helpers in `src/models/pyro_model.py` via `build_guide_effects` and `compute_linear_predictor`.
Implemented steps 5–7 in `src/models/pyro_model.py` inside `fate_model`: priors, random-walk gene effects, guide deviations, softmax mapping, and soft-label likelihood.
Implemented steps 8–10 in `src/models/pyro_model.py`: minibatched SVI via `fit_svi` (uses `pyro.plate` subsampling) and primary contrast helper `primary_contrast_mes_ec`.
Implemented step 11 in `src/models/pyro_model.py`: `reconstruct_theta_samples` now draws posterior theta samples from the guide and `export_gene_summary_for_ash` computes weighted MES–EC summaries (betahat, sebetahat) and writes the ash-ready CSV.
Added day-weight columns (`w0..w{D-1}`) to the gene summary CSV output for traceability.
Implemented step 12 in pipeline form: added `scripts/fit_pyro_export.py`, `scripts/run_ash.R`, and `scripts/rank_hits.py`, plus `config.yaml`, `Snakefile`, and conda env specs in `envs/`.
Implemented step 13 diagnostics in `scripts/run_diagnostics.py` (holdout cross-entropy for full vs nuisance-only model; optional guide permutation with ash if enabled).
Added shared IO helpers in `scripts/pyro_io.py` and a `scripts/__init__.py` package marker.
Remaining stubs: `load_inputs`, `run_ash_rscript`, `main` in `src/models/pyro_pipeline.py`.
