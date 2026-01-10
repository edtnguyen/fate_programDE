## Mod4 spec (consistent, θ-only biology): entropy/day-weighted likelihood + principled sigma_time + GPU-parallel θ bootstrap + NTC reference-day calibration

Scope: You want **gene KD affects MES fate** ⇒ estimand is **θ_{gene, MES, day}** (MES–EC contrast). We will (i) downweight unreliable CellRank cells/days, (ii) regularize time evolution without forcing signs, and (iii) get uncertainty from **cell bootstrap refits** (θ-level SE + sign probability). NTC guides are used for **calibration/diagnostics**, not for calling.

---

# 1) Config (`config.yaml`) — single source of truth

Add/replace these keys:

```yaml
# ---------- Reliability weighting (CellRank trust) ----------
reliability_weighting:
  enable: true

  # per-cell confidence from fate-prob entropy
  entropy:
    enable: true
    w_min: 0.20            # floor on per-cell weight
    eps: 1.0e-12           # avoid log(0)
    normalize_mean_to_1: true

  # per-day lambda derived from a reference cell set (uses NTCs)
  day_lambda:
    enable: true
    method: "median_conf"  # median_conf | mean_conf
    ref: "max_conf"        # max_conf | last_day
    monotone_increasing: true   # optional but recommended if later days more reliable

    reference_cells:
      mode: "ntc_only"     # ntc_only | low_target_burden | all
      low_target_burden_max: 1  # used if mode=low_target_burden

# ---------- Time smoothing (θ random walk) ----------
time_smoothing:
  enable: true
  D: 4                     # days (d0..d3), needed for interval indexing

  sigma_time:
    base_scale: 0.20       # same role as old s_time; start around your current
    hierarchical: true
    global_scale_prior: 0.20   # HalfNormal for s_time_global per fate contrast

    # optionally adapt interval smoothness using day lambdas
    interval_scale_from_day_lambda: true
    interval_scale_min: 0.30
    interval_scale_max: 1.50

# ---------- Bootstrap (θ only) ----------
bootstrap:
  enable: true
  B: 50
  seed: 12345
  stratify_by: ["day", "rep"]

  # per-bootstrap fit controls
  svi_steps: 6000
  posterior_draws_for_point: 200     # compute θ point estimate from draws; 0 => use VI loc
  output_level: "theta"              # fixed in Mod4

  # scheduling
  max_concurrent_gpus: 8

# ---------- Calling from bootstrap summaries ----------
call_theta_bootstrap:
  per_day_q: 0.05
  effect_size_eps: 0.20
  sign_prob_thresh: 0.90

# ---------- Paths ----------
paths:
  outdir: "out_fate_pipeline"
  outdir_bootstrap: "out_fate_pipeline/bootstrap"

# ---------- Diagnostics ----------
diagnostics:
  write_weights_summary: true
  write_ntc_diagnostics: true
```

---

# 2) Data definitions (consistent with your model)

* Days: `d0..d3` (D=4)
* Fate contrast: MES vs EC reference.
* θ target: `theta[gene, MES, day]`
* CellRank fate probs: `p_i = (p_EC, p_MES, p_NEU)`.

Guide/cell structure unchanged.

---

# 3) Reliability weighting implementation

## 3.1 Per-cell entropy weight `w_i`

For each cell i:

* (H_i=-\sum_f p_{if}\log(p_{if}+\varepsilon))
* (H_{\max}=\log(3))
* `conf_i = 1 - H_i/H_max`  (0..1)
* `w_i = w_min + (1-w_min)*conf_i`

## 3.2 Reference cells for day λ (NTC usage)

We estimate day reliability λ_d from a reference set to avoid perturbation-driven bias.

Reference cell mask rules:

* `ntc_only`: cells where **no non-NTC guides are present** (after k<=Kmax filtering).
  (Implementation: count nonzero among non-NTC guide columns per cell == 0.)
* `low_target_burden`: cells with `#nonNTC_guides <= low_target_burden_max`
* `all`: all cells

## 3.3 Compute day lambdas λ_d

On reference cells only:

* per day d: `c_d = median(conf_i | day=d)` (or mean if configured)
* pick a reference day `c_ref`:

  * `max_conf`: `c_ref = max_d c_d`
  * `last_day`: `c_ref = c_{d3}`
* `λ_d_raw = c_d / c_ref`

If `monotone_increasing=true`:

* enforce non-decreasing λ across day index using isotonic regression.

## 3.4 Combine into final per-cell weight used in likelihood

For cell i:

* `weight_i = w_i * λ_{day[i]}`

If `normalize_mean_to_1=true`:

* `weight_i /= mean(weight_i)` (computed once on all cells or large subsample)

## 3.5 Apply weights in the model likelihood

Your soft-label multinomial likelihood:

* `ll_i = sum_f p_if * log_softmax(eta_i)[f]`

Replace with:

* `ll_i_weighted = weight_i * ll_i`
* `pyro.factor("ll", (N/batch) * ll_i_weighted.sum())`

No other scaling changes.

## 3.6 Diagnostics output

Write `{outdir}/weights_summary.json`:

* per day: `median_entropy`, `median_conf`, `lambda_d`
* overall: quantiles of `weight_i` and `w_i`

Also write `{outdir}/ntc_diagnostics.json` (optional):

* counts of reference cells per day
* entropy/conf distributions for reference vs all cells

---

# 4) Time smoothing: principled sigma_time (θ RW)

Current RW1:

* `theta[d] = theta[d-1] + sigma_time[interval] * eps`

Intervals:

* j=0: d0→d1
* j=1: d1→d2
* j=2: d2→d3

## 4.1 Hierarchical shrinkage on sigma_time

Introduce per contrast f* (MES):

* `s_time_global ~ HalfNormal(global_scale_prior)`
* `sigma_time[j] ~ HalfNormal(sigma_prior_scale_j)`

Where base:

* `sigma_prior_scale_j = base_scale * s_time_global * interval_multiplier_j`

## 4.2 Interval multiplier from day lambdas (optional but recommended)

If enabled:

* `r_j = 0.5*(λ_{d=j} + λ_{d=j+1})`
* `interval_multiplier_j = clip(r_j, interval_scale_min, interval_scale_max)`

This automatically tightens intervals adjacent to low-confidence days without hardcoding “day2”.

## 4.3 What not to do

* Do not reverse-walk unless you have a modeling reason; with the above, uncertainty in early days is handled by λ and sigma_time priors.
* Do not force signs; let bootstrap decide stability.

---

# 5) Bootstrap uncertainty (θ-only) — GPU-parallel via Snakemake

## 5.1 Bootstrap resampling scheme (stratified)

For each bootstrap replicate b:

* For each stratum (day d, rep r):

  * sample with replacement `n_{d,r}` cells from that stratum’s original indices
* concatenate sampled indices → bootstrap dataset
* keep original filtering rules (k<=Kmax etc.)

This preserves day/rep structure.

## 5.2 Per-bootstrap fit command-line interface

Update `scripts/fit_model.py` (or your runner) to accept:

* `--bootstrap_id INT` (0..B-1)
* `--bootstrap_seed INT`
* `--svi_steps INT`
* `--posterior_draws_for_point INT`
* `--out_prefix PATH` (directory)
* `--use_gpu 1`

Behavior:

* if `bootstrap_id` provided, resample indices using `bootstrap_seed` and `stratify_by`.
* compute weights (entropy + λ) on the bootstrap dataset (λ computed from reference cells within bootstrap).
* run SVI for `svi_steps`.
* export θ point estimates.

## 5.3 Output schema per bootstrap replicate

Write: `{outdir_bootstrap}/b{b}/theta_point_estimates.csv`

Required columns:

* `gene`
* `betahat_d0`, `betahat_d1`, `betahat_d2`, `betahat_d3`  (θ MES contrast)

Recommended columns:

* `fit_method` = `vi_draw_mean` if posterior_draws_for_point>0 else `vi_loc`
* `lambda_d0..d3` actually used
* `w_mean`, `w_p10`, `w_p50`, `w_p90`
* `n_cells_d0..d3`
* `converged` (1/0)

Also write `{outdir_bootstrap}/b{b}/fit_meta.json` (seed, steps, etc.)

## 5.4 Bootstrap summarizer (new)

Create `scripts/bootstrap_summarize_theta.py`

Inputs:

* `{outdir_bootstrap}/b*/theta_point_estimates.csv`
* `call_theta_bootstrap` thresholds

Compute per gene×day:

* `median_betahat_d*`
* `se_boot_d*` = sample SD across bootstraps (ddof=1)
* `sign_prob_pos_d*` = fraction of bootstraps with betahat>0
* `z_d* = median / se_boot`
* `p_d*` from normal approx (two-sided)
* `q_d*` BH FDR within each day

Outputs:

1. `{outdir_bootstrap}/bootstrap_theta_summary.csv`
2. `{outdir_bootstrap}/bootstrap_hits_daywise.csv` (criteria below)
3. `{outdir_bootstrap}/bootstrap_hits_anyday.csv` (optional, but consistent)

Calling criteria (daywise):

* `q_d < per_day_q`
* `abs(median_betahat_d) >= effect_size_eps`
* sign stability: `sign_prob_pos_d >= sign_prob_thresh` OR `<= 1-sign_prob_thresh`

Any-day (optional):

* compute min p across days and apply Bonferroni or require daywise criteria on at least one day.

---

# 6) Snakemake changes (GPU parallel on Sherlock)

## 6.1 Rules

Add in `Snakefile`:

* `bootstrap_fit` (GPU)
* `bootstrap_summarize_theta` (CPU)
* `bootstrap_all` (phony target)

Use snakemake resources:

* `resources: gpu=1` for `bootstrap_fit`
* run with `--resources gpu=<max_concurrent_gpus>`

Example:

```python
B = int(config["bootstrap"]["B"])
BOOTDIR = config["paths"]["outdir_bootstrap"]

rule bootstrap_fit:
    output:
        est = f"{BOOTDIR}/b{{b}}/theta_point_estimates.csv",
        meta = f"{BOOTDIR}/b{{b}}/fit_meta.json",
    resources:
        gpu=1,
        mem_mb=24000,
        time_min=180
    threads: 4
    shell:
        r"""
        mkdir -p {BOOTDIR}/b{wildcards.b}
        python scripts/fit_model.py \
          --config config.yaml \
          --bootstrap_id {wildcards.b} \
          --bootstrap_seed {config[bootstrap][seed]} \
          --svi_steps {config[bootstrap][svi_steps]} \
          --posterior_draws_for_point {config[bootstrap][posterior_draws_for_point]} \
          --out_prefix {BOOTDIR}/b{wildcards.b} \
          --use_gpu 1
        """
```

(Implementation detail: seed should be `base_seed + b`; enforce that in code.)

Summarize:

```python
rule bootstrap_summarize_theta:
    input:
        expand(f"{BOOTDIR}/b{{b}}/theta_point_estimates.csv", b=range(B))
    output:
        summary = f"{BOOTDIR}/bootstrap_theta_summary.csv",
        hits_day = f"{BOOTDIR}/bootstrap_hits_daywise.csv",
        hits_any = f"{BOOTDIR}/bootstrap_hits_anyday.csv",
    resources:
        mem_mb=8000,
        time_min=30
    threads: 2
    shell:
        r"""
        python scripts/bootstrap_summarize_theta.py \
          --config config.yaml \
          --bootstrap_dir {BOOTDIR} \
          --B {B} \
          --out_summary {output.summary} \
          --out_hits_day {output.hits_day} \
          --out_hits_any {output.hits_any}
        """
```

## 6.2 Sherlock GPU partition

In your slurm profile (or rule-level cluster params), ensure bootstrap_fit uses:

* `--partition=gpu`
* `--gres=gpu:1`

Then run:

```bash
snakemake --profile profiles/sherlock -j 200 --resources gpu=8 bootstrap_all
```

---

# 7) NTC guides — what we do (and what we don’t)

## 7.1 What Mod4 uses NTC for

1. Define reference cells (`ntc_only`) to estimate λ_d (day reliability).
2. Diagnostics: compare entropy/conf distributions between reference and all cells by day.

## 7.2 What Mod4 does NOT do

* We do not use NTC to set calling thresholds for θ directly.
* We do not force θ trajectories to match NTC behavior.
* We do not try to “fix” TAOK1 sign by controls; bootstrap determines stability.

---

# 8) Acceptance tests (must pass)

1. `weights_summary.json` exists and λ differs by day (unless truly identical).
2. Bootstrap outputs:

* B files exist
* `theta_point_estimates.csv` has required columns, no NaNs
* `bootstrap_theta_summary.csv` has se_boot > 0 for most genes

3. For your Hippo day2:

* TAOK1: expect `sign_prob_pos_d2` near 0 if it’s robustly negative
* SAV1: expect `sign_prob_pos_d2` closer to 0.5 if it’s weak/noisy

---

# 9) Deliverables checklist (files)

Modify:

* `config.yaml`
* `scripts/fit_model.py` (bootstrap + weights)
* Pyro model module (apply weight_i in likelihood; sigma_time priors)
* `Snakefile`

Add:

* `scripts/cellrank_weights.py`
* `scripts/bootstrap_summarize_theta.py`
* `scripts/ntc_diagnostics.py` (optional but recommended)

Outputs:

* `{outdir}/weights_summary.json`
* `{outdir}/ntc_diagnostics.json`
* `{outdir_bootstrap}/b*/theta_point_estimates.csv`
* `{outdir_bootstrap}/bootstrap_theta_summary.csv`
* `{outdir_bootstrap}/bootstrap_hits_daywise.csv`
* `{outdir_bootstrap}/bootstrap_hits_anyday.csv`

This is internally consistent with “θ-only gene biology” and with GPU-parallel execution via Snakemake on Sherlock.

