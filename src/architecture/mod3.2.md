````markdown
# CODEX SPEC v3.3: θ-based gene biology discovery + probabilistic “meaningfully large” calling (MES) + eval alignment

## Goal
Upgrade the θ-based gene-biology discovery path to use a **probabilistic magnitude gate** instead of a hard `abs(postmean) >= eps`, while keeping:
- day-resolved activity (`active_days`)
- conservative lfsr control
- simulation evaluation aligned to θ truth

Core calling becomes:

For each gene/day with mash posterior approx `θ_d ~ Normal(μ_d, s_d^2)`:
- sign confidence: `lfsr_d < t_day`
- practical magnitude: `P(|θ_d| > eps) >= pmin`

This avoids borderline mean-threshold artifacts and stabilizes calls under uncertainty.

---

## Definitions
### Inputs (MES–EC)
From `mash_gene_<mode>.csv` (θ-based):
- `postmean_d0..d3` = μ_d
- `postsd_d0..d3` = s_d
- `lfsr_d0..d3`

Config parameters:
- `effect_size_eps = 0.20`   (MES; practical threshold on logit scale)
- `effect_prob_large_min = 0.90` (default; allow 0.95 stricter)
- `mash_lfsr_thresh_day = 0.05`
- any-day correction: Bonferroni → `t_any = mash_lfsr_thresh_day / D` unless overridden

### Probability-of-large computation
For each day d:
- `p_small_d = P(|θ_d| <= eps) = Φ((eps - μ_d)/s_d) - Φ((-eps - μ_d)/s_d)`
- `p_large_d = 1 - p_small_d`

Numerical requirements:
- clamp `s_d = max(s_d, 1e-8)` before dividing
- `Φ` is standard Normal CDF

### Probabilistic call for day activity
- `active_day_d = (lfsr_d < mash_lfsr_thresh_day) AND (p_large_d >= effect_prob_large_min)`

### Probabilistic call for any-day hits (Bonferroni)
Let D=4 and:
- if `mash_lfsr_thresh_anyday` is null:
  - `t_any = mash_lfsr_thresh_day / D` (0.0125)
- else `t_any = mash_lfsr_thresh_anyday`

Then:
- `hit_anyday = any_d (lfsr_d < t_any AND p_large_d >= effect_prob_large_min)`

Optional (extra FP brake; OFF by default):
- `min_active_days_for_hit_anyday` (default 1)
- `hit_anyday = (count_d active_anyday_day_d >= min_active_days_for_hit_anyday)`

---

## 1) Config changes (`config.yaml`)
Add/override:

```yaml
# MES practical magnitude threshold (logit scale)
effect_size_eps: 0.20

# NEW: probabilistic magnitude gating
effect_prob_large_min: 0.90        # recommend 0.90; try 0.95 for stricter

# lfsr thresholds
mash_lfsr_thresh_day: 0.05
mash_anyday_correction: bonferroni
mash_lfsr_thresh_anyday: null      # default 0.05/4=0.0125

# discovery source-of-truth
discovery_source: "gene_theta"

# optional extra brake
min_active_days_for_hit_anyday: 1  # keep at 1 unless you want stronger FP control
````

Backward compatibility:

* If `effect_prob_large_min` is missing, default it in code to 0.90.

---

## 2) Pipeline wiring (Snakemake): canonical = mash_gene

### Requirement

Final `hits_ranked.csv` MUST come from `mash_gene_{mash_default_mode}.csv`.

Do not use:

* `gene_from_guide_mash_*` for discovery
* `mash_guide_*` for discovery

Keep them as QC outputs only if desired.

---

## 3) Update `scripts/rank_hits.py` (primary work item)

### 3.1 Inputs

Add CLI args (or reuse existing):

* `--in-mash-gene OUT/mash_gene_<mode>.csv` (required for gene_theta)
* `--config config.yaml`
* `--out OUT/hits_ranked.csv`

The script must:

* read config keys above
* infer D from columns (expect d0..d3)

### 3.2 Implement Normal CDF

Preferred:

* `from scipy.stats import norm; norm.cdf(x)`
  Fallback (no scipy):
* `cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))` using `math.erf`

### 3.3 Compute p_large per day

For each gene row and day d:

Inputs:

* `mu = postmean_dd`
* `sd = postsd_dd` (clamp >= 1e-8)
* `eps = effect_size_eps`

Compute:

* `z1 = (eps - mu) / sd`
* `z0 = (-eps - mu) / sd`
* `p_small = cdf(z1) - cdf(z0)`
* `p_large = 1.0 - p_small`
  Clamp p_large into [0,1] (numerical).

Create new columns:

* `plarge_d0..plarge_d3`
* `max_plarge` (max over days)
* `min_plarge` (optional)

### 3.4 Active day logic

Compute:

* `active_d = (lfsr_d < t_day) AND (plarge_d >= pmin)`
  where:
* `t_day = mash_lfsr_thresh_day` (0.05)
* `pmin = effect_prob_large_min` (0.90)

Generate:

* boolean columns `active_d0..active_d3`
* `active_days` string: join days where active
* `n_active_days`

Compute:

* `best_day = argmin_d lfsr_d` (over all days)
* `best_day_among_active = argmin_{d where active_d} lfsr_d` else NA
* `min_lfsr_among_active = min_{d where active_d} lfsr_d` else +inf
* `max_abs_postmean = max_d abs(postmean_d)`

### 3.5 Any-day logic (Bonferroni)

Determine t_any:

* if cfg.mash_lfsr_thresh_anyday is not null → use it
* else if cfg.mash_anyday_correction == "bonferroni" → `t_any = t_day / D`
* else `t_any = t_day`

Compute per day:

* `active_any_d = (lfsr_d < t_any) AND (plarge_d >= pmin)`
* `n_any_active = sum(active_any_d)`
  Then:
* `hit_anyday = (n_any_active >= cfg.min_active_days_for_hit_anyday)`

Add:

* `active_days_any` (string)
* `n_any_active`

### 3.6 Row filtering and ranking

The output `hits_ranked.csv` should include *all genes* by default, but sorted with hits first.
(If you prefer only hits, add a flag `--only-hits`.)

Sort keys:

1. `hit_anyday` desc
2. `min_lfsr_among_active` asc
3. `max_abs_postmean` desc

### 3.7 Output columns

Include at minimum:

* `gene`
* `hit_anyday`
* `active_days` / `n_active_days`
* `active_days_any` / `n_any_active`
* `best_day_among_active`
* `min_lfsr_among_active`
* daywise: `postmean_d*`, `postsd_d*`, `lfsr_d*`, `plarge_d*`

---

## 4) Update simulation evaluation script to use probabilistic calling (θ truth)

Locate the script that writes `daywise_confusion.csv`. Modify predicted positives to match the new rule.

### 4.1 Predicted active gene/day

Use mash-gene posterior summary:

* `pred_active(g,d) = (lfsr_d < t_day) AND (p_large_d >= pmin)`

### 4.2 Truth active gene/day

Truth is θ-only:

* `true_active(g,d) = (abs(theta_true_d) >= eps)`
  Where:
* `eps = effect_size_eps` (0.20)

### 4.3 Any-day confusion

Predicted:

* `pred_any(g) = any_d (lfsr_d < t_any AND p_large_d >= pmin)`
  Truth:
* `true_any(g) = any_d (abs(theta_true_d) >= eps)`

Compute TP/FP/FN/TN exactly as before.

### 4.4 Outputs

Write:

* `OUT/daywise_confusion.csv` with rows d0..d3, overall, any and columns tp/fp/fn/tn.

Also (NEW, for debugging) write:

* `OUT/pred_call_summary.json` with:

  * distribution of `n_any_active`
  * number of predicted hits
  * mean/max of plarge among called days

---

## 5) QC: keep guide-level E1 but label it QC-only

No change required, but ensure:

* `gene_from_guide_mash_<mode>.csv` is not used for `hits_ranked.csv` nor eval.

Optionally:

* write `OUT/qc_guide_heterogeneity.csv` later.

---

## 6) Acceptance criteria

1. `hits_ranked.csv` is generated from `mash_gene_<default_mode>.csv` and includes `plarge_d*`.
2. Day activity is determined by `(lfsr < 0.05) AND (plarge >= 0.90)`.
3. Any-day uses Bonferroni lfsr threshold (0.0125) unless overridden.
4. Simulation confusion numbers change consistently when toggling `effect_prob_large_min` (0.90 vs 0.95).
5. Evaluation uses θ truth and predicted calls from mash_gene (not guide aggregation).

---

## 7) Implementation checklist

1. Update `config.yaml` (Section 1).
2. Patch `Snakefile` so `rank_hits.py` consumes `mash_gene_{mash_default_mode}.csv`.
3. Update `rank_hits.py` to compute `plarge_d*` and probabilistic calling.
4. Patch eval script generating `daywise_confusion.csv` to use probabilistic calling + θ truth.
5. Run:

   * `python -m py_compile scripts/rank_hits.py <eval_script>.py`
   * `sim_all` and compare confusion vs hard threshold variant
   * `perm_all` (optional) should remain ~0 hits under conservative settings

```
```
