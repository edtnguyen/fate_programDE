fate_programDE
==============================

Quantify DE effects of CRISPRi gene KD on fate probabilities

Model overview
------------

This pipeline models how CRISPRi guide content in high-MOI cells shifts CellRank
fate probabilities across three terminal fates (EC, MES, NEU). Key points:

- Observations: per-cell fate probabilities `p_i` (simplex over EC/MES/NEU),
  day `d_i`, replicate `r_i`, and a padded list of guides per cell
  (`guide_ids[i, m]`, `mask[i, m]`) with `Kmax` maximum guides.
- Preprocessing: cells with `k_i > Kmax` are dropped; NTC guides map to
  `guide_id=0` (real entry, `mask=1`), and padding uses `guide_id=0` with
  `mask=0`. Guide burden `k_i` is centered within day.
- Linear predictor (non-reference fates): day intercepts `alpha`, replicate
  effects `b`, guide-burden slope `gamma * k_i`, plus a summed guide effect for
  each cell that accounts for multi-guide cells (MOI).
- Guide effects decompose hierarchically:
  `beta_{g,f*,d} = theta_{gene(g),f*,d} + delta_{g,f*}` where
  `theta` is the gene-by-day trajectory and `delta` is a guide-specific deviation.
- Time smoothing: gene effects follow a fate-specific random walk with scale
  `sigma_time`. We additionally support a fixed per-interval scale
  `time_scale` (length `D-1`) so later-day increments can be larger if later
  time points show more heterogeneity. Example for `D=4`:
  `time_scale: [1.0, 1.3, 1.6]`.
- Priors: HalfNormal priors on `tau`, `sigma_time`, `sigma_guide`, and
  Gaussian priors on nuisance effects `alpha`, `b`, and `gamma`.
- Fitting: stochastic variational inference (SVI) with minibatches.
- Reporting: gene-level MES–EC effects summarized across days with weighted
  averages, followed by adaptive shrinkage (ash) to compute lfsr/q-values.

Pipeline and usage
------------

Primary entrypoints:

- Fit + export: `scripts/fit_pyro_export.py`
- Shrinkage: `scripts/run_ash.R`
- Rank hits: `scripts/rank_hits.py`
- Diagnostics: `scripts/run_diagnostics.py`
- Simulation: `scripts/simulate_recovery.py`
- Snakemake: `Snakefile` (real data and simulation targets)

Configuration:

- `config.yaml` controls input paths, model sizes, priors, and diagnostics.
- `time_scale` (optional) must have length `D-1` and positive entries.

Required inputs:

- `adata_path`: AnnData with
  - `adata.obsm[fate_prob_key]` containing EC/MES/NEU probabilities.
  - `adata.obsm[guide_key]` containing the guide-by-cell matrix.
  - `adata.obsm[covar_key]` with `rep_key` for replicate labels.
  - `adata.obs[day_key]` with day labels (e.g., d0–d3).
- `guide_map_csv`: CSV with columns `guide_name`, `gene_name`, `is_ntc`.

End-to-end on real data (uses `config.yaml`):

```bash
snakemake --use-conda --cores 1
```

This runs: `fit_pyro_export` → `run_ash` → `rank_hits` to produce
`out_fate_pipeline/hits_ranked.csv`.

Simulation (full chain, including ash + ranking):

```bash
snakemake --use-conda --cores 1 sim_all
```

The simulation settings live in `config.yaml` under `sim_*` keys. You can
override scenarios by editing:

- `sim_cells`, `sim_genes`, `sim_guides`, `sim_days`, `sim_reps`, `sim_kmax`
- `sim_concentration`, `sim_num_steps`, `sim_num_draws`
- `sim_sweep_*` and `sim_tau_sweep_*` for prior sweeps

To disable the forced rerun of simulation, set:

```
sim_always_run: false
```

Run simulation directly (no Snakemake):

```bash
python scripts/simulate_recovery.py \
  --cells 200 \
  --genes 30 \
  --guides 90 \
  --days 4 \
  --kmax 4 \
  --concentration 50 \
  --num-steps 500 \
  --s-time 0.3 \
  --s-guide 0.5 \
  --s-tau 1.0 \
  --time-scale 1.0,1.3,1.6 \
  --run-export \
  --write-anndata \
  --force
```

Common simulation knobs:

- `--concentration`: lower values increase fate-probability noise.
- `--cells` and `--num-steps`: control sample size and optimization effort.
- `--s-time`, `--s-guide`, `--s-tau`: prior scales.
- `--time-scale`: per-interval random-walk scaling (length `days-1`).

Tests
------------

Unit tests (core helpers):

```bash
python -m unittest tests/test_pyro_model_helpers.py
```

Integration tests (fit + export flow):

```bash
python -m unittest tests/test_pyro_model_integration.py
```

Full test suite (environment check + all tests):

```bash
PYTHON_BIN=python scripts/run_tests.sh
```

Optional end-to-end logic check:

```bash
python scripts/test_pyro_model_logic.py
```

Diagnostics
------------

```bash
python scripts/run_diagnostics.py \
  --config config.yaml \
  --adata data/adata.h5ad \
  --guide-map data/guide_map.csv \
  --out out_fate_pipeline/diagnostics.json
```

HPC (Sherlock / Slurm-style) notes
------------

On Sherlock (or any Slurm cluster), run Snakemake with a cluster submission
wrapper. Example (adjust partition/account/time/memory to your environment):

```bash
snakemake --use-conda --jobs 50 --latency-wait 60 \
  --cluster "sbatch -p <partition> -c {threads} --mem {resources.mem_mb}"
```

Tips:

- Keep `--use-conda` to isolate Python/R dependencies per rule.
- Increase `--jobs` and `--cores` based on your allocation.
- If your filesystem is slow, add `--latency-wait`.
- GPU rules already declare `resources: gpu=1` in the Snakefile; adjust the
  `--cluster` string to request GPUs if needed.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

