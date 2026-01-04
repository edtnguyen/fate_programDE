````markdown
# CODEX SPEC v3: Reduce FPs in mash-based discovery (effect-size floor + meta combine + V regularization + SE calibration + conservative default + permutation calibration)

## Goal
Modify the existing Pyro→export→mashr(two-mode)→aggregate→rank pipeline to reduce false positives (FPs) in simulated (and real) runs while preserving day-resolved interpretability.

Core decisions:
- Turn on an effect-size floor: `effect_size_eps: 0.15`.
- Keep day-resolved reporting: explicitly output which day(s) a gene is active (`active_days`) rather than collapsing to a single min-lfsr decision.
- Replace guide→gene aggregation from `min_lfsr` to a principled fixed-effect meta combine (inverse-variance; Stouffer-equivalent when z are Normal).
- Replace “V fallback to identity when singular” with a principled, stable regularization (nearPD + eigen floor + shrink-to-identity).
- Add SE inflation and SE floor before mash.
- Require `mash_min_strong_for_cov >= 50` before PCA/ED covariances.
- Set `mash_default_mode: "conservative"`.
- Add an optional permutation-calibration pipeline branch to tune SE inflation/floor and thresholds.

Non-goals:
- Do NOT change the Pyro likelihood/model structure (soft-label multinomial logistic, embedding-sum guide list, guide→gene hierarchy, time RW, burden/day/rep covariates).
- Do NOT add sparsity priors (no horseshoe/spike).
- Do NOT integrate rpy2; keep R called via Snakemake.

## Current files (already present)
- `pyro_model.md`: model spec (unchanged).
- `mod2_discovery_improv.md`: current discovery improvements spec; update it conceptually but implement changes in code/config.
- `scripts/run_mashr_two_mode.R`: mash runner (modify).
- `scripts/aggregate_guides_to_genes.py` (may already exist per v2 plan; modify heavily to meta combine).
- Snakemake rules (modify to wire new columns/steps).
- Exporter `scripts/fit_pyro_export.py` (already outputs gene/guide betahat+se; keep; no model change). Only confirm it exports correct daywise columns.

---

## 1) Config changes (`config.yaml`)
Update/add these keys (keep existing keys intact; only add/override):

```yaml
# ---- hit calling / interpretability ----
effect_size_eps: 0.15

# day-level calling (still keep mash_lfsr_thresh for backwards compat)
mash_lfsr_thresh_day: 0.05

# optional correction for "any-day" selection (recommended default ON)
mash_anyday_correction: "bonferroni"   # one of: "none", "bonferroni"
# if bonferroni, threshold used for declaring hit_anyday (not for per-day display)
# default behavior: mash_lfsr_thresh_anyday = mash_lfsr_thresh_day / D
mash_lfsr_thresh_anyday: null          # if null, compute as mash_lfsr_thresh_day / 4

# ---- mash mode selection ----
mash_modes: ["conservative", "enriched"]
mash_default_mode: "conservative"

# ---- mash: SE calibration (apply to Shat before mash_set_data) ----
mash_se_inflate: 1.50
mash_se_floor: 0.02

# ---- mash: V estimation + regularization ----
mash_estimate_V: true
mash_v_eig_floor: 1e-6       # used in eigen clamp after nearPD
mash_v_shrink_rho: 0.05      # shrink-to-identity weight in [0,1]

# ---- mash: data-driven covariances gating ----
mash_add_datadriven_cov: true
mash_strong_lfsr_for_cov: 0.1
mash_cov_pca_npc: 2
mash_min_strong_for_cov: 50  # require >=50 "strong" rows before cov_pca/cov_ed

# ---- enriched mode knobs (kept, but now less permissive by default) ----
mash_enriched_prior: "uniform"
mash_enriched_nullweight: 3
mash_enriched_usepointmass: true

# ---- guide->gene aggregation method ----
guide_to_gene_agg_method: "meta_fixed"   # new default
````

Notes:

* Keep existing `mash_lfsr_thresh` for compatibility, but update all downstream logic to use `mash_lfsr_thresh_day` and `mash_lfsr_thresh_anyday` instead.
* For D=4 days, if `mash_lfsr_thresh_anyday` is null, compute `0.05/4 = 0.0125`.

---

## 2) Modify `scripts/run_mashr_two_mode.R`

### 2.1 Add SE inflation + SE floor before mash_set_data

Currently:

* Reads `Bhat`, `Shat` and immediately does `data <- mash_set_data(Bhat, Shat)`.

Change:

* After reading `Bhat`, `Shat`, apply:

  * `Shat <- Shat * cfg$mash_se_inflate` (default 1.0 if missing)
  * `Shat <- pmax(Shat, cfg$mash_se_floor)` (default 0.0 if missing)

Implementation snippet to insert right after `Shat <- as.matrix(...)`:

```r
se_inflate <- if (!is.null(cfg$mash_se_inflate)) cfg$mash_se_inflate else 1.0
se_floor   <- if (!is.null(cfg$mash_se_floor))   cfg$mash_se_floor   else 0.0

Shat <- Shat * se_inflate
if (se_floor > 0) Shat <- pmax(Shat, se_floor)

data <- mash_set_data(Bhat, Shat)
```

### 2.2 Replace "V is singular -> identity" with principled regularization fallback

Current behavior:

* Estimate V via `estimate_null_correlation_simple(data)`.
* If eigenvalues invalid or <= eig_floor: set `Vhat <- I`.

Replace with:

* Always build a valid correlation matrix via:

  1. `nearPD(Vhat, corr=TRUE, keepDiag=TRUE)` (Matrix package)
  2. eigenvalue clamp to `>= mash_v_eig_floor`
  3. re-normalize to correlation (diag=1)
  4. shrink-to-identity with `rho = mash_v_shrink_rho` (default 0.05)
* If `estimate_null_correlation_simple` errors: start from identity and still run the regularizer (which will keep it identity).

Required R dependency:

* add `library(Matrix)` in suppressPackageStartupMessages block.

Drop-in helper function (place near top of script after cfg load):

```r
regularize_V <- function(V, eig_floor = 1e-6, rho = 0.05) {
  # nearest PSD correlation, then eigen floor, then shrink-to-I
  Vnpd <- as.matrix(nearPD(V, corr = TRUE, keepDiag = TRUE)$mat)

  eig <- eigen(Vnpd, symmetric = TRUE)
  vals <- pmax(eig$values, eig_floor)
  Vreg <- eig$vectors %*% diag(vals) %*% t(eig$vectors)

  # renormalize to correlation
  Dinv <- diag(1 / sqrt(diag(Vreg)))
  Vreg <- Dinv %*% Vreg %*% Dinv

  # shrink to identity
  K <- ncol(Vreg)
  Vout <- (1 - rho) * Vreg + rho * diag(K)
  return(Vout)
}
```

Then after estimating `Vhat` (or identity), do:

```r
eig_floor <- if (!is.null(cfg$mash_v_eig_floor)) cfg$mash_v_eig_floor else 1e-6
rho <- if (!is.null(cfg$mash_v_shrink_rho)) cfg$mash_v_shrink_rho else 0.05
Vhat <- regularize_V(Vhat, eig_floor = eig_floor, rho = rho)

data <- mash_set_data(Bhat, Shat, V = Vhat)
```

Remove the old “if eigen <= eig_floor then identity” block entirely.

### 2.3 Gate PCA/ED covariances on `mash_min_strong_for_cov`

Current behavior:

* In enriched mode: if `length(strong_idx) >= 2` then add `cov_pca/cov_ed`.

Change:

* Use `min_strong <- cfg$mash_min_strong_for_cov` (default 50).
* Only add data-driven covariances if `length(strong_idx) >= min_strong`.

Replace the `if (length(strong_idx) >= 2)` check with:

```r
min_strong <- if (!is.null(cfg$mash_min_strong_for_cov)) cfg$mash_min_strong_for_cov else 50
if (length(strong_idx) >= min_strong) {
  ...
} else {
  message(sprintf("Warning: strong set too small for data-driven covariances (n=%d < %d); using canonical only.", length(strong_idx), min_strong))
}
```

### 2.4 Output posterior SD per day (`postsd_d*`)

Current behavior:

* Computes `postmean <- get_pm(m)` and `lfsr <- get_lfsr(m)`
* Writes `postmean_d*` and `lfsr_d*`.

Change:

* Also compute `postsd <- get_psd(m)` and write `postsd_d*` columns aligned to days.

Insert:

```r
postsd <- get_psd(m)
```

And inside the per-day loop:

```r
dt[[paste0("postsd_d", day)]] <- postsd[, j]
```

Keep existing `lfsr_min` and `best_day` calculations.

### 2.5 Keep conservative as the default operational mode

No code change needed beyond config; ensure Snakemake uses `mash_default_mode` when selecting final outputs.

---

## 3) Update/implement `scripts/aggregate_guides_to_genes.py` (meta combine)

Goal: convert guide-level mash output to gene-level daywise posterior (mean/sd), daywise lfsr, and hit calls that:

* preserve day interpretability (`active_days`),
* reduce FPs (effect-size floor + no min-over-guides),
* optionally apply any-day correction (bonferroni) only for `hit_anyday`.

### 3.1 Input schema (guide mash output CSV)

Expect columns (per guide row):

* `guide`, `gene`
* `postmean_d0..postmean_d3`
* `postsd_d0..postsd_d3`   (NEW)
* `lfsr_d0..lfsr_d3`       (keep, but aggregator will recompute gene lfsr from meta posterior)

### 3.2 Meta combine per gene/day (fixed-effect inverse-variance)

For each gene ℓ and day d:

* For each guide g targeting ℓ:

  * μ_gd = postmean_gd
  * s_gd = postsd_gd (clamp to >= tiny to avoid div-by-zero; e.g. 1e-8)
* weights: w_gd = 1 / s_gd^2
* gene posterior mean:

  * μ_ℓd = (Σ_g w_gd μ_gd) / (Σ_g w_gd)
* gene posterior sd:

  * s_ℓd = sqrt( 1 / Σ_g w_gd )

### 3.3 Compute gene/day lfsr from Normal posterior

Assume gene/day posterior is approximately Normal N(μ_ℓd, s_ℓd^2).

Compute:

* p_neg = P(effect < 0) = Φ((0 - μ)/s)
* p_pos = 1 - p_neg
* lfsr_ℓd = min(p_neg, p_pos)

Implementation: use `scipy.stats.norm.cdf` or implement via `math.erf`.

### 3.4 Day-level activity + reporting (this answers “which day(s)?”)

Config:

* `effect_size_eps` (0.15)
* `mash_lfsr_thresh_day` (0.05)

Define:

* `hit_day_d = (lfsr_ℓd < mash_lfsr_thresh_day) AND (abs(μ_ℓd) >= effect_size_eps)`
* `active_days = comma-joined list of days where hit_day_d is true` (e.g. `"d1,d3"`; empty string if none)
* `best_day = argmin_d lfsr_ℓd` (over all days)
* `best_day_among_active = argmin_{d where hit_day_d} lfsr_ℓd` else NA

### 3.5 Define `hit_anyday` with optional any-day correction

Config:

* `mash_anyday_correction`: `"none"` or `"bonferroni"`
* `mash_lfsr_thresh_anyday`: if null, compute `mash_lfsr_thresh_day / D` (D inferred from number of day columns, should be 4)

Define:

* If correction == `"none"`:

  * `hit_anyday = any_d(hit_day_d)`
* If correction == `"bonferroni"`:

  * Let `t_any = mash_lfsr_thresh_anyday` (default 0.0125)
  * Define a stricter day-test for `hit_anyday` only:

    * `hit_day_any_d = (lfsr_ℓd < t_any) AND (abs(μ_ℓd) >= effect_size_eps)`
  * `hit_anyday = any_d(hit_day_any_d)`
  * Still report `active_days` using the *day threshold* (0.05) for interpretability (so you see biology), but `hit_anyday` uses the corrected threshold to reduce FPs.

### 3.6 Output schema (gene-level CSV)

Write `OUT/gene_from_guide_mash_<mode>.csv` with columns:

Required:

* `gene`
* `n_guides`
* `gene_postmean_d0..d3`
* `gene_postsd_d0..d3`
* `gene_lfsr_d0..d3`
* `best_day`
* `best_day_among_active`
* `active_days`
* `hit_anyday`

Diagnostics (recommended):

* `max_abs_postmean` = max_d |gene_postmean_dd|
* `min_lfsr` = min_d gene_lfsr_dd
* `min_lfsr_among_active` (NA if no active)
* `guide_min_lfsr_min` = min over guides of (min_d lfsr_gd) (for debugging only; not used for calling)
* `median_guide_lfsr_min`

### 3.7 CLI

Implement:

```bash
python scripts/aggregate_guides_to_genes.py \
  --in-mash-guide OUT/mash_guide_<mode>.csv \
  --out-gene OUT/gene_from_guide_mash_<mode>.csv \
  --config config.yaml
```

---

## 4) Update ranking script (if present) to use new columns

If you have `scripts/rank_hits.py`:

* Input should default to `gene_from_guide_mash_{mash_default_mode}.csv`
* Ranking: primarily by `min_lfsr_among_active` (ascending), then by `max_abs_postmean` (descending).
* Display columns:

  * `gene`, `hit_anyday`, `active_days`, `best_day_among_active`, `min_lfsr_among_active`, `max_abs_postmean`, plus daywise postmean/lfsr.

Output:

* `OUT/hits_ranked.csv`

---

## 5) Snakemake updates

### 5.1 Ensure mash runner produces postsd columns

* `run_mashr_two_mode.R` now outputs `postsd_d*` columns in addition to existing `postmean_d*` and `lfsr_d*`.
* Downstream aggregator depends on `postsd_d*`.

### 5.2 Wire two-mode mash for guides (required) and genes (optional)

For each `mode` in `config["mash_modes"]`:

* `OUT/mash_guide_<mode>.csv` from `OUT/guide_daywise_for_mash.csv`
* `OUT/gene_from_guide_mash_<mode>.csv` from `OUT/mash_guide_<mode>.csv`

Optionally keep gene mash outputs:

* `OUT/mash_gene_<mode>.csv` from `OUT/gene_daywise_for_mash.csv` (not required for calling if guide-mash is primary).

### 5.3 Default mode selection

Final `hits_ranked.csv` should use:

* `OUT/gene_from_guide_mash_{config[mash_default_mode]}.csv`

### 5.4 Add mode comparison

Keep/extend `OUT/mash_mode_comparison.csv`:

* number of hits in each mode
* overlap / Jaccard
* median `min_lfsr_among_active`
* (optional) distribution summaries of `active_days` counts

---

## 6) Add permutation calibration branch (optional but recommended)

Goal: run the *full* pipeline once on a null dataset created by permuting guide assignments within strata to estimate FP calibration and tune:

* `mash_se_inflate`, `mash_se_floor`
* `mash_lfsr_thresh_day` / `mash_lfsr_thresh_anyday`
* possibly confirm `effect_size_eps: 0.15` is appropriate

### 6.1 Permutation definition (data-level; before Pyro fit)

Create a permuted dataset by shuffling `guide_ids` across cells within each stratum:

* stratum = (day, rep, k_bin)
* k_bin can be exact k, or bins like [1-2], [3-5], [6-10], [11-20] (choose one and document)
* Permute entire padded guide list (the multiset of guides per cell) so burden and MOI structure is preserved.

Implement a script (new):

* `scripts/make_permuted_guides.py`
  Inputs:
* preprocessed tensors or adata-derived exported arrays (`guide_ids`, `mask`, `day`, `rep`, `k`)
  Outputs:
* `guide_ids_perm`, `mask_perm` (mask unchanged), same shape

Then the permuted run uses the same Pyro training code but reads permuted guides.

### 6.2 Pipeline wiring

Add a Snakemake target:

* `OUT_PERM/hits_ranked.csv` computed by:

  * permute guides
  * fit Pyro
  * export
  * mash (conservative)
  * aggregate meta
  * rank

Also add:

* `OUT_PERM/perm_summary.json` with:

  * total hits
  * hits by number of active days
  * max |postmean| distribution for called vs not
  * suggested threshold adjustments (informational only)

---

## 7) Acceptance tests (must pass)

1. `run_mashr_two_mode.R`:

   * Outputs `postmean_d*`, `postsd_d*`, `lfsr_d*`, `lfsr_min`, `best_day` for both modes.
   * Never hard-falls back to identity V due to singularity; instead regularizes V (identity only if estimation fails outright).
2. Data-driven covariances:

   * Only added when `n_strong >= mash_min_strong_for_cov` (default 50).
3. Aggregator:

   * Produces daywise gene posterior mean/sd/lfsr via meta combine.
   * Produces `active_days`, `best_day_among_active`, and `hit_anyday` (with optional bonferroni correction).
   * Uses effect-size floor `effect_size_eps=0.15`.
4. Default mode:

   * `mash_default_mode` is conservative; final `hits_ranked.csv` uses conservative guide-mash aggregation.
5. Permutation branch (if enabled):

   * Produces near-zero hits under conservative defaults; if not, increase `mash_se_inflate` / `mash_se_floor` or tighten any-day threshold.

---

## 8) Implementation checklist (ordered)

1. Update `config.yaml` with new keys and defaults (effect_size_eps=0.15; default mode conservative; SE/V knobs; strong set gate; meta_fixed agg).
2. Patch `scripts/run_mashr_two_mode.R`:

   * add Matrix dependency
   * apply `mash_se_inflate` and `mash_se_floor`
   * implement `regularize_V()` and remove identity fallback-on-eigen check
   * gate PCA/ED covariances with `mash_min_strong_for_cov`
   * output `postsd_d*`
3. Implement/modify `scripts/aggregate_guides_to_genes.py`:

   * add `meta_fixed` method (default)
   * compute gene/day posterior mean/sd and gene/day lfsr
   * implement `active_days` and corrected `hit_anyday`
   * write new columns
4. Update `rank_hits.py` (if exists) to rank using `min_lfsr_among_active` + `max_abs_postmean`, and to display `active_days`.
5. Update `Snakefile` to wire:

   * mash guide outputs (two modes)
   * aggregator per mode
   * final hits from default mode
   * comparison table
6. (Optional) Add permutation calibration scripts + Snakemake branch.

---

## Notes for Codex

* Prefer minimal diff patches: change only what is required.
* Do not rename existing output files; only add new columns and new optional outputs.
* Ensure numeric stability:

  * clamp postsd to >= 1e-8 in meta weights
  * handle genes with 1 guide (meta still works)
  * handle missing days robustly (but expect d0..d3 present)
* Keep R failures safe:

  * if mash fails with provided covariances, keep existing retry behavior, but still use regularized V.

```
```
