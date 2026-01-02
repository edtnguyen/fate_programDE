fate_programDE
==============================

Quantify Differential Effect (DE) of CRISPRi-based gene KD on fate probabilities

Model
------------

We model how CRISPRi guide content in high-MOI cells shifts CellRank fate
probabilities across three terminal fates (EC, MES, NEU). EC is the reference
fate, and $f^\star \in \{\mathrm{MES}, \mathrm{NEU}\}$ denotes non-reference
fates.

Observed data and indexing:

- Cells $i=1,\dots,N$ with fate probabilities
  $p_i=(p_{i,\mathrm{EC}},p_{i,\mathrm{MES}},p_{i,\mathrm{NEU}})$ and
  $\sum_f p_{i,f}=1$.
- Day $d_i \in \{0,\dots,D-1\}$ and replicate $r_i \in \{0,\dots,R-1\}$.
- Guides $g=1,\dots,G$ (non-NTC) and genes $\ell=1,\dots,L$ with map $\ell(g)$.
- Padded guide lists: $\mathrm{guide\_ids}_{i,m}$ and $\mathrm{mask}_{i,m}$ for $m=1,\dots,K_{\max}$.
- NTC guides map to $g=0$ (real entry, $\mathrm{mask}=1$); padding uses $g=0$
  with $\mathrm{mask}=0$.
- Guide burden $k_i = \sum_m \mathrm{mask}_{i,m}$, centered within day:
  $\tilde{k}_i = k_i - \bar{k}_{d_i}$.

Guide effect decomposition:

$$
\beta_{g,f^\star,d} = \theta_{\ell(g),f^\star,d} + \delta_{g,f^\star},\quad g \ge 1
$$

with hard-zero baselines:

$$
\theta_{0,f^\star,d} = 0,\quad \delta_{0,f^\star} = 0,\quad \beta_{0,f^\star,d}=0.
$$

Linear predictor (MOI-aware sum over guides):

$$
\eta_{i,f^\star} =
\alpha_{f^\star,d_i}
+ b_{f^\star,r_i}
+ \gamma_{f^\star}\tilde{k}_i
+ \sum_{m=1}^{K_{\max}} \mathrm{mask}_{i,m}\,\beta_{\mathrm{guide\_ids}_{i,m},f^\star,d_i}.
$$

Softmax mapping to probabilities:

$$
\eta_{i,\mathrm{EC}}=0,\quad \pi_i = \mathrm{softmax}(\eta_i).
$$

Likelihood (soft-label cross-entropy):

$$
\log L = \sum_{i=1}^N \sum_{f\in\{\mathrm{EC,MES,NEU}\}} p_{i,f}\,\log \pi_{i,f}.
$$

Priors (weakly regularizing):

$$
\sigma_{\alpha,f^\star} \sim \mathrm{HalfNormal}(s_\alpha),\quad
\alpha_{f^\star,d} \sim \mathcal{N}(0,\sigma_{\alpha,f^\star}^2)
$$

$$
\sigma_{\mathrm{rep},f^\star} \sim \mathrm{HalfNormal}(s_{\mathrm{rep}}),\quad
b_{f^\star,r} \sim \mathcal{N}(0,\sigma_{\mathrm{rep},f^\star}^2)
$$

$$
\sigma_{\gamma,f^\star} \sim \mathrm{HalfNormal}(s_\gamma),\quad
\gamma_{f^\star} \sim \mathcal{N}(0,\sigma_{\gamma,f^\star}^2)
$$

Gene effects (random walk with optional fixed time scale):

$$
\tau_{f^\star} \sim \mathrm{HalfNormal}(s_\tau),\quad
z_{\ell,f^\star} \sim \mathcal{N}(0,1),\quad
\theta_{\ell,f^\star,0} = \tau_{f^\star} z_{\ell,f^\star}
$$

$$
\sigma_{\mathrm{time},f^\star} \sim \mathrm{HalfNormal}(s_{\mathrm{time}}),\quad
\varepsilon_{\ell,f^\star,d} \sim \mathcal{N}(0,1)
$$

$$
\theta_{\ell,f^\star,d} =
\theta_{\ell,f^\star,d-1} +
c_d\,\sigma_{\mathrm{time},f^\star}\,\varepsilon_{\ell,f^\star,d},\quad d=1,\dots,D-1.
$$

Here $c_d$ is a fixed per-interval scale from `time_scale` (length $D-1$). If
`time_scale` is omitted, we use $c_d=1$ for all intervals.

Guide deviations:

$$
\sigma_{\mathrm{guide},f^\star} \sim \mathrm{HalfNormal}(s_{\mathrm{guide}}),\quad
u_{g,f^\star} \sim \mathcal{N}(0,1),\quad
\delta_{g,f^\star} = \sigma_{\mathrm{guide},f^\star} u_{g,f^\star}.
$$

Fitting and reporting:

- Fit with SVI (minibatches over cells).
- Define day-specific MES–EC gene effects $\Delta_\ell(d)=\theta_{\ell,\mathrm{MES},d}$.
- Summarize across days with weights $w_d$ (default $w_d \propto n_d$):

$$
\Delta^{\mathrm{sum}}_\ell = \sum_{d=0}^{D-1} w_d\,\Delta_\ell(d),\quad \sum_d w_d = 1.
$$

- Report posterior mean and uncertainty for $\Delta^{\mathrm{sum}}_\ell$ and apply
  adaptive shrinkage (ash) to compute lfsr/q-values for hit calling.

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

    ├── Snakefile          <- Snakemake workflow (real + simulation targets)
    ├── config.yaml        <- Main config for real-data runs
    ├── envs/              <- Conda environments (pyro.yaml, ash.yaml)
    ├── scripts/           <- Pipeline entrypoints + utilities
    │   ├── fit_pyro_export.py
    │   ├── run_ash.R
    │   ├── rank_hits.py
    │   ├── run_diagnostics.py
    │   ├── simulate_recovery.py
    │   ├── simulate_prior_sweep.py
    │   ├── simulate_tau_sweep.py
    │   ├── simulate_stress.py
    │   └── run_tests.sh
    ├── src/
    │   ├── models/        <- Pyro model + helpers
    │   │   ├── pyro_model.py
    │   │   └── pyro_pipeline.py
    │   └── architecture/  <- Model/pipeline specs + notes
    │       ├── pyro_context.md
    │       ├── pyro_model.md
    │       ├── pyro_export_skeleton.md
    │       ├── pyro_pipeline_skeleton.md
    │       ├── snakefile.md
    │       └── memory.md
    ├── tests/             <- Unit + integration tests
    ├── data/              <- Input data (adata, guide map) + raw/interim/processed
    ├── out_fate_pipeline/ <- Real-data outputs
    ├── out_fate_pipeline_sim/ <- Simulation outputs
    ├── docs/              <- Sphinx docs (optional)
    ├── notebooks/         <- Jupyter notebooks
    ├── references/        <- Data dictionaries and manuals
    ├── reports/           <- Figures + reports
    ├── requirements.txt   <- Python deps for local use
    ├── setup.py
    ├── README.md
    ├── LICENSE
    └── tox.ini
