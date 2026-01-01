Added Pyro pipeline/model stubs based on `src/architecture/pyro_pipeline_skeleton.md`, `src/architecture/pyro_context.md`, and `src/architecture/pyro_model.md`.
Implemented observed-data preprocessing (step 1) in `src/models/pyro_pipeline.py`: `build_padded_guides`, `filter_cells_by_k`, `make_k_centered`, and `to_torch` now follow the skeleton with basic validation.
Implemented latent indexing metadata (step 2) in `src/models/pyro_model.py` via `LatentIndex` plus fate name constants, exposing parameter shapes and fate index mappings.
Implemented guide-level decomposition (step 3) and linear predictor (step 4) helpers in `src/models/pyro_model.py` via `build_guide_effects` and `compute_linear_predictor`.
Implemented steps 5–7 in `src/models/pyro_model.py` inside `fate_model`: priors, random-walk gene effects, guide deviations, softmax mapping, and soft-label likelihood.
Implemented steps 8–10 in `src/models/pyro_model.py`: minibatched SVI via `fit_svi` (uses `pyro.plate` subsampling) and primary contrast helper `primary_contrast_mes_ec`.
Remaining stubs: `load_inputs`, `run_ash_rscript`, `main` in `src/models/pyro_pipeline.py`, and `reconstruct_theta_samples`, `export_gene_summary_for_ash` in `src/models/pyro_model.py`.
