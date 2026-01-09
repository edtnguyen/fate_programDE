fate_programDE
==============================

Quantify Differential Effect (DE) of CRISPRi-based gene KD on fate probabilities

Setup
------------

Install the repo in editable mode (from the repo root) so `src` imports resolve:

```bash
pip install -e .
```


Model (full specification)
------------

**1. Observed data (after filtering)**

- Cells $i=1,\dots,N$ after filtering to $k_i \le K_{\max}$, where $k_i$ is the
  number of detected guides in cell $i$ ($K_{\max}=20$ in current runs).
- Days $d_i \in \{0,\dots,D-1\}$ and replicates $r_i \in \{0,\dots,R-1\}$
  ($D=4$ for d0--d3; $R=2$ in current data).
- CellRank fate probabilities
  $p_i=(p_{i,\mathrm{EC}},p_{i,\mathrm{MES}},p_{i,\mathrm{NEU}})$ with
  $\sum_f p_{i,f}=1$.
- Guides $g=1,\dots,G$ (non-NTC), genes $\ell=1,\dots,L$, and map $\ell(g)$
  (current data: $G=2000$, $L=300$).
- Embedding-sum representation per cell: define $g_{i,m}$ and $m_{i,m}$ for
  $m=1,\dots,K_{\max}$, where $g_{i,m}$ is `guide_ids[i,m]` and $m_{i,m}$ is
  `mask[i,m]` with $m_{i,m}\in\{0,1\}$ indicating real vs padding.
- Hard-zero convention: NTC guides map to $g=0$ (real entry, $m_{i,m}=1$);
  padding uses $g=0$ with $m_{i,m}=0$. Define `gene_of_guide[0]=0` and
  `gene_of_guide[g]=\ell(g)` for $g\ge 1$.
- Guide burden:

$$
k_i = \sum_{m=1}^{K_{\max}} m_{i,m}.
$$

- Centered guide burden:

$$
\tilde{k}_i = k_i - \bar{k}_{d_i}.
$$

**2. Latent parameters and indexing**

- Reference fate: EC. Non-reference fates:
  $\mathcal{F}^\star = \{\mathrm{MES}, \mathrm{NEU}\}$, indexed by $f^\star$.
- Gene-by-day effects $\theta_{\ell,f^\star,d}$ and guide deviations
  $\delta_{g,f^\star}$ (time-invariant, shared across days).
- Mean-zero constraint within each gene: for each gene $\ell$ and fate $f^\star$,
  $\frac{1}{|G_\ell|}\sum_{g\in G_\ell}\delta_{g,f^\star}=0$ (enforced by centering).
- Nuisance: day intercepts $\alpha_{f^\star,d}$, replicate effects
  $b_{f^\star,r}$, burden slope $\gamma_{f^\star}$.
- Hard-zero baseline rows: $\theta_{0,f^\star,d}=0$ and $\delta_{0,f^\star}=0$.

**3. Guide-level effect decomposition**

$$
\beta_{g,f^\star,d} = \theta_{\ell(g),f^\star,d} + \delta_{g,f^\star},\quad g \ge 1
$$

with $\beta_{0,f^\star,d}=0$ for NTC/baseline guides.

Because $\delta$ is centered within each gene, the guide-mean effect equals the
gene effect: $\frac{1}{|G_\ell|}\sum_{g\in G_\ell}\beta_{g,f^\star,d}=\theta_{\ell,f^\star,d}$.

**4. Cell-level linear predictor (MOI-aware)**

$$ \eta _{i,f^\star} = \alpha _{f^\star,d _i} + b _{f^\star,r _i} + \gamma _{f^\star} \tilde{k} _i + \sum _{m=1}^{K _{\max}} m _{i,m} \beta _{g _{i,m}, f^\star, d _i}, \qquad \eta _{i,\mathrm{EC}}=0. $$

Let $G_i$ denote the set of guides detected in cell $i$; the padded sum above
is equivalent to $\sum_{g\in G_i}\beta_{g,f^\star,d_i}$.

This embedding-sum is the MOI correction: each guide's effect is estimated
conditional on other guides co-occurring in the same cell.

**5. Softmax mapping**

$$
\pi_i = \mathrm{softmax}(\eta_i),\quad
\pi_{i,f}=\frac{\exp(\eta_{i,f})}{\sum_{f'\in\{\mathrm{EC,MES,NEU}\}}\exp(\eta_{i,f'})}.
$$

**6. Likelihood (soft-label cross-entropy)**

$$
\log L = \sum_{i=1}^N \sum_{f\in\{\mathrm{EC,MES,NEU}\}} p_{i,f}\,\log \pi_{i,f}.
$$

**7. Priors (fixed-scale, weakly regularizing)**

Day/rep/burden effects:

$$
\alpha_{f^\star,d}\sim \mathcal{N}(0, s_\alpha),\quad
b_{f^\star,r}\sim \mathcal{N}(0, s_{\mathrm{rep}}),\quad
\gamma_{f^\star}\sim \mathcal{N}(0, s_\gamma).
$$

Gene effects (random walk with per-interval scales):

$$
\tau_{f^\star}\sim \mathrm{HalfNormal}(s_\tau),\quad
z_{\ell,f^\star,0}\sim \mathcal{N}(0,1),\quad
\theta_{\ell,f^\star,0} = \tau_{f^\star} z_{\ell,f^\star,0}.
$$

For intervals $d=1,\dots,D-1$:

$$
\sigma_{\mathrm{time},f^\star,d}\sim \mathrm{HalfNormal}(s_{\mathrm{time}}),\quad
\varepsilon_{\ell,f^\star,d}\sim \mathcal{N}(0,1),
$$

$$
\theta_{\ell,f^\star,d} =
\theta_{\ell,f^\star,d-1} +
\sigma_{\mathrm{time},f^\star,d}\,\varepsilon_{\ell,f^\star,d}.
$$

Guide deviations (mean-zero within gene):

$$
\sigma_{\mathrm{guide},f^\star}\sim \mathrm{HalfNormal}(s_{\mathrm{guide}}),\quad
u_{g,f^\star}\sim \mathcal{N}(0,1),\quad
\delta^{\mathrm{raw}}_{g,f^\star}= \sigma_{\mathrm{guide},f^\star} u_{g,f^\star}.
$$

$$
\delta_{g,f^\star} = \delta^{\mathrm{raw}}_{g,f^\star} -
\frac{1}{|G_{\ell(g)}|}\sum_{g'\in G_{\ell(g)}} \delta^{\mathrm{raw}}_{g',f^\star}.
$$

**8. Non-centered parameterization**

We fit non-centered latents $(z,\varepsilon,u)$ and scales
$(\tau,\sigma_{\mathrm{time}},\sigma_{\mathrm{guide}})$, then deterministically
construct $(\theta,\delta)$. This improves SVI stability.

**9. SVI training objective with minibatching**

For minibatch $S$ of size $B$:

$$
\log p(p\mid\phi) \approx w\frac{N}{B}\sum_{i\in S}\sum_f p_{i,f}\log \pi_{i,f}(\phi).
$$

Implementation uses `pyro.plate("cells", N, subsample_size=B)` and scales the
log-likelihood by $N/B$ with `ClippedAdam`; $w$ is `likelihood_weight`.

**10. Primary contrast (MES-EC)**

$$
\Delta_\ell(d) \equiv \theta_{\ell,\mathrm{MES},d}.
$$

**11. Posterior summaries + exports (theta/beta)**

Draw $S$ posterior samples from the VI guide. For the MES contrast:

- Gene/day summaries (theta only):
```math
\hat{\theta}_{\ell,d}=\frac{1}{S}\sum_{s=1}^{S}\theta^{(s)}_{\ell,\mathrm{MES},d}
````

and posterior SD $\hat{\sigma}_{\ell,d}$ computed across draws.

* Guide/day summaries (beta):

```math
\beta^{(s)}_{g,d}=\theta^{(s)}_{\ell(g),\mathrm{MES},d}+\delta^{(s)}_{g,\mathrm{MES}}
```

with mean and posterior SD across draws.

Exports:

- `gene_daywise_for_mash.csv`: `gene`, `betahat_d*`, `se_d*` from $\theta$ only.
- `guide_daywise_for_mash.csv`: `guide`, `gene`, `betahat_d*`, `se_d*` from $\beta$.
- `theta_posterior_summary.npz` and `delta_posterior_summary.npz`.
- QC: `qc_delta_mean_by_gene.csv` (mean $\delta$ per gene) and
  `qc_theta_beta_offset_by_gene.csv` (mean over guides of $\beta$ minus $\theta$
  across days), both expected near 0.

Optional across-day summary (legacy ash):

$$
\Delta^{\mathrm{sum}}_\ell = \sum_{d=0}^{D-1} w_d\,\Delta_\ell(d),
$$

with $w_d\ge 0$ and $\sum_d w_d=1$ (default $w_d \propto n_d$).

**12. Empirical Bayes shrinkage + hit calling (mashr two-mode + aggregation)**

- Run mashr on `gene_daywise_for_mash.csv` and `guide_daywise_for_mash.csv` in
  two modes (conservative/enriched).
- Aggregate guide-level mash outputs to gene-level meta-fixed estimates
  (`gene_from_guide_mash_*.csv`) with daywise postmean/postsd/lfsr and
  `active_days`.
- Default discovery uses the gene-theta mash output (`mash_gene_*.csv`).

Daywise call at day $d$ if:

- `lfsr_d < mash_lfsr_thresh_day`, and

```math
  P(|\mathrm{effect}_d|\ge \epsilon)\ge \mathrm{effect\_prob\_large\_min}
```
  where the tail probability is computed from a Normal(postmean, postsd).

Any-day call uses the configured correction (bonferroni by default) and
requires `min_active_days_for_hit_anyday` active days.

**13. Minimal diagnostics (recommended)**

- Held-out cross-entropy: full model vs nuisance-only.
- Negative control: permute guides within (day, rep, k-bin) and confirm hits collapse.
- Sanity checks: known regulators show expected MES-EC direction.

Pipeline and usage
------------

Primary entrypoints:

- Fit + export: `scripts/fit_pyro_export.py`
- Shrinkage (mashr two-mode): `scripts/run_mashr_two_mode.R`
- Guide→gene aggregation: `scripts/aggregate_guides_to_genes.py`
- Mode comparison: `scripts/compare_mash_modes.py`
- Rank hits: `scripts/rank_hits.py`
- Diagnostics: `scripts/run_diagnostics.py`
- Simulation: `scripts/simulate_recovery.py`
- Snakemake: `Snakefile` (real data and simulation targets)

Configuration:

- `config.yaml` controls input paths, model sizes, priors, and diagnostics.
- `time_scale` is deprecated in the current model (per-interval $\sigma_{\mathrm{time}}$ is learned).

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

Using existing conda envs on an HPC (skip YAML env creation): edit the env
paths at the top of `Snakefile`:

```python
PYRO_ENV = "/oak/stanford/groups/engreitz/Users/tri/envs/sc-dl-gpu"
R_ENV = "/oak/stanford/groups/engreitz/Users/tri/envs/scrnaR"
```

Then run Snakemake normally with `--use-conda` (it will activate those envs):

```bash
snakemake --use-conda --cores 1
```

To submit to Slurm and use the per-rule `mem_mb` values from the Snakefile:

```bash
snakemake --use-conda --jobs 50 \
  --cluster "sbatch -p {resources.partition} -t {resources.time} -c {threads} --mem={resources.mem_mb}"
```

Override resources for specific rules on the CLI (example: more memory/time for
`fit_pyro_export` and more threads):

```bash
snakemake --use-conda --jobs 50 \
  --cluster "sbatch -p {resources.partition} -t {resources.time} -c {threads} --mem={resources.mem_mb}" \
  --set-resources fit_pyro_export:mem_mb=200000,fit_pyro_export:time=12:00:00 \
  --set-threads fit_pyro_export=8
```

This runs: `fit_pyro_export` → mashr (gene + guide, both modes) → guide aggregation →
`rank_hits` (default mode) and produces `out_fate_pipeline/hits_ranked.csv`
plus `out_fate_pipeline/mash_mode_comparison.csv`.

Simulation (full chain, including mashr + aggregation + ranking):

```bash
snakemake --use-conda --cores 1 sim_all
```

The simulation settings live in `config.yaml` under `sim_*` keys. You can
override scenarios by editing:

- `sim_cells`, `sim_genes`, `sim_guides`, `sim_days`, `sim_reps`, `sim_kmax`
- `sim_concentration`, `sim_num_steps`, `sim_num_draws`
- `sim_sweep_*` and `sim_tau_sweep_*` for prior sweeps

Each simulation run writes `out_fate_pipeline_sim/sim_metadata.yaml` with the
exact inputs used (cells/genes/guides, concentration, seed, day counts, etc.).
Override the path with `sim_metadata_path` in `config.yaml`.

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
  --metadata-out out_fate_pipeline_sim/sim_metadata.yaml \
  --run-export \
  --write-anndata \
  --force
```

Common simulation knobs:

- `--concentration`: lower values increase fate-probability noise.
- `--cells` and `--num-steps`: control sample size and optimization effort.
- `--s-time`, `--s-guide`, `--s-tau`: prior scales.
- `--time-scale`: per-interval random-walk scaling (length `days-1`).

Daywise recovery evaluation (sim outputs):

```bash
python scripts/eval_daywise_hits.py \
  --hits out_fate_pipeline_sim/hits_ranked.csv \
  --truth out_fate_pipeline_sim/sim_recovery.csv \
  --lfsr-thresh 0.05 \
  --truth-thresh 0.2
```

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
    ├── envs/              <- Conda environments (pyro.yaml, ash.yaml, mash.yaml)
    ├── scripts/           <- Pipeline entrypoints + utilities
    │   ├── fit_pyro_export.py
    │   ├── run_ash.R
    │   ├── run_mashr.R
    │   ├── run_mashr_two_mode.R
    │   ├── aggregate_guides_to_genes.py
    │   ├── compare_mash_modes.py
    │   ├── rank_hits.py
    │   ├── run_diagnostics.py
    │   ├── eval_daywise_hits.py
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
    ├── out_fate_pipeline_sim/ <- Simulation outputs (incl. sim_metadata.yaml)
    ├── docs/              <- Sphinx docs (optional)
    ├── notebooks/         <- Jupyter notebooks
    ├── references/        <- Data dictionaries and manuals
    ├── reports/           <- Figures + reports
    ├── requirements.txt   <- Python deps for local use
    ├── setup.py
    ├── README.md
    ├── LICENSE
    └── tox.ini
