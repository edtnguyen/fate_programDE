````markdown
# CODEX SPEC v2: Improve discovery without new data (mashr deconservatize + E1 guide-level mash + two-mode mash script)

## Goal
Discovery is low. Implement **safe power improvements** without collecting more data by:
A) making mashr less conservative for a **pre-screened enriched panel**,
B) adding **data-driven covariances** in mashr,
C) providing a **two-mode mashr runner** (conservative vs enriched) for sensitivity,
D) implementing **E1: guide-level mashr** (≈2000 rows) to increase effective panel size, then aggregating back to genes.

Keep the Pyro model structure unchanged (soft-label multinomial, embedding-sum, guide→gene hierarchy, burden/day/rep covariates, random walk on gene-by-day effects, time-invariant guide deviations).

---

## Summary of required changes
1) Modify exporter to output *both*:
   - gene-level daywise MES effect: `theta_mes[gene, day]`
   - guide-level daywise MES effect: `beta_mes[guide, day] = theta_mes[gene(g), day] + delta_g`
   with posterior mean and posterior SD.

2) Add `scripts/run_mashr_two_mode.R` that can run in two modes:
   - **mode=conservative**: current/default behavior (nullbiased, nullweight=10, point mass)
   - **mode=enriched**: less conservative (prior="uniform", nullweight=1, optional usepointmass FALSE)
   and optionally adds data-driven covariances (PCA/ED) in enriched mode.

3) Add `scripts/aggregate_guides_to_genes.py` (or extend `rank_hits.py`) to:
   - take mashr output on guides and map to genes (6 guides/gene, variable)
   - produce a gene-level table with "hit_anyday" based on guide evidence.

4) Update Snakemake to:
   - run mashr on genes (optional)
   - run mashr on guides (recommended default)
   - run aggregator to get final `hits_ranked.csv`
   - run both mash modes and emit comparison outputs.

---

## Inputs (no change)
- `adata.obs["day"]` -> day labels (4)
- `adata.obsm["covar"]["rep"]` -> replicate labels (2)
- `adata.obsm["guide"]` -> guide presence matrix (N x G, sparse preferred)
- `guide_map_csv` mapping guide -> gene, and which guides are NTC

---

## Outputs (new/modified)
### Pyro exporter outputs
Write two CSVs:

1) `OUT/gene_daywise_for_mash.csv`
Columns:
- `gene`
- `betahat_d0 ... betahat_d3`  (MES logit only; EC reference)
- `se_d0 ... se_d3`

2) `OUT/guide_daywise_for_mash.csv`
Columns:
- `guide` (guide ID or name)
- `gene`  (mapped gene name)
- `betahat_d0 ... betahat_d3`  (MES logit guide-level effect)
- `se_d0 ... se_d3`

### mashr outputs
For each input csv, produce:
- `OUT/mash_gene_<mode>.csv`
- `OUT/mash_guide_<mode>.csv`

They include `postmean_d*`, `lfsr_d*`, `lfsr_min`, `best_day`.

### Aggregation output
- `OUT/gene_from_guide_mash_<mode>.csv` (gene-level aggregated from guide mashr)
- `OUT/hits_ranked.csv` (final ranking uses enriched+guide by default)

### Comparison output
- `OUT/mash_mode_comparison.csv` summarizing:
  - number of hits (genes) by mode
  - overlap Jaccard
  - top-50 overlap
  - median lfsr_min

---

## Config additions (`config.yaml`)
Add:
```yaml
mash_modes: ["conservative", "enriched"]
mash_default_mode: "enriched"

# mash knobs (enriched)
mash_enriched_prior: "uniform"       # mash() prior argument
mash_enriched_nullweight: 1          # reduce null favoring
mash_enriched_usepointmass: true     # try true first; can set false in config
mash_add_datadriven_cov: true
mash_cov_pca_npc: 2
mash_strong_lfsr_for_cov: 0.1        # or strong set based on 1by1
mash_estimate_V: true                # else use identity

# hit thresholds
mash_lfsr_thresh: 0.05               # used for gene hit_anyday
guide_to_gene_agg_method: "min_lfsr" # or "stouffer"
effect_size_eps: 0.0                 # optional: require max|postmean| > eps
````

Keep existing config keys intact.

---

## Detailed implementation steps

# 1) Modify `scripts/fit_pyro_export.py` exporter

### 1.1 Keep gene-daywise exporter (already exists)

Ensure it writes `gene_daywise_for_mash.csv` as described.

### 1.2 Add guide-daywise exporter (E1)

Implement function:

* Draw posterior samples from guide:
  `samples = guide.sample_posterior([S], *model_args)`

* Reconstruct `theta` and `delta` samples:

  * `theta_samples`: [S, L, Fstar, D]
  * `delta_samples`: [S, G_internal, Fstar] (time-invariant)

* Construct guide-level effects:
  For each guide g, with gene index l(g):
  [
  \beta_{g,f*,d}^{(s)} = \theta_{l(g),f*,d}^{(s)} + \delta_{g,f*}^{(s)}
  ]
  For MES dimension only:

  * `beta_mes_samples`: [S, G_internal, D]

* Compute:

  * `betahat = mean_s(beta_mes_samples)`
  * `se = std_s(beta_mes_samples, ddof=1)`

* Write CSV with guide name/id and gene name:

  * include NTC guides too (optional), but if included their betahat should be ~0.

IMPORTANT:

* Exclude baseline/padding guide_id=0 if it is hard-zero and not a real guide name.
* Use the `guide_map_csv` to map internal guide_id -> guide_name and gene_name.

### 1.3 CLI output behavior

`fit_pyro_export.py` should accept:

* `--out-gene` path
* `--out-guide` path
  (Keep `--out` for backward compatibility but not required.)

---

# 2) Add mash runner: `scripts/run_mashr_two_mode.R`

This script runs mashr for an input CSV with daywise betahat/se columns.

## CLI

`Rscript scripts/run_mashr_two_mode.R <in_csv> <out_csv> <mode> <config_yaml>`

* mode in {"conservative","enriched"}.

## Common steps

* read CSV
* build `Bhat`, `Shat`
* set `Vhat`:

  * if config `mash_estimate_V`: `estimate_null_correlation_simple(data)` (note: can be unstable for enriched panels)
  * else: `diag(D)`
* `data <- mash_set_data(Bhat, Shat, V=Vhat)`

## Mode=conservative

Use mash defaults:

* `prior = "nullbiased"` (default)
* `nullweight = 10` (default)  (keep explicit for reproducibility)
* `usepointmass = TRUE` (default)
* covariance set: `cov_canonical(data)` only

## Mode=enriched

Read knobs from config:

* `prior = mash_enriched_prior` (default "uniform")
* `nullweight = mash_enriched_nullweight` (default 1)
* `usepointmass = mash_enriched_usepointmass` (default TRUE)

Covariances:

* Always include canonical: `U.c <- cov_canonical(data)`
* If `mash_add_datadriven_cov`:

  * compute `m.1by1 <- mash_1by1(data)`
  * pick "strong set" indices:

    * option A: `get_significant_results(m.1by1, thresh=0.1)`  (or config)
    * option B: use lfsr threshold on 1by1
  * subset data.strong
  * `U.pca <- cov_pca(data.strong, npc=mash_cov_pca_npc)`
  * `U.ed <- cov_ed(data.strong, U.pca)`
  * set `Ulist <- c(U.c, U.ed)`
  * else `Ulist <- U.c`

Fit:

* `m <- mash(data, Ulist, prior=..., nullweight=..., usepointmass=...)`

Output:

* add `postmean_d*`, `lfsr_d*`, `lfsr_min`, `best_day`
* preserve input columns
* write `out_csv`

---

# 3) Add guide->gene aggregator: `scripts/aggregate_guides_to_genes.py`

Purpose: convert guide-level mash output to gene-level hit calls.

## Input

* mash guide output CSV (contains guide, gene, postmean_d*, lfsr_d*)
* config.yaml for thresholds and method

## Output

* gene-level CSV with columns:

  * `gene`
  * `lfsr_min` (gene-level)
  * `best_day`
  * optionally `postmean_bestday`
  * `hit_anyday` based on gene-level lfsr_min < mash_lfsr_thresh and optional effect size criterion
  * plus diagnostics: `n_guides`, `min_guide_lfsr_min`, `median_guide_lfsr_min`

## Aggregation methods (config `guide_to_gene_agg_method`)

### Method 1: "min_lfsr" (default; simple, powerful, may be optimistic)

* For each guide, compute `guide_lfsr_min = min_d lfsr_{g,d}`
* For gene:

  * `gene_lfsr_min = min_{g in gene} guide_lfsr_min`
  * `best_day` inherited from the guide/day achieving min

### Method 2: "stouffer" (more conservative)

* Convert per-guide best-day z using sign of postmean and lfsr->approx p:

  * p ≈ lfsr (approx, acceptable heuristic) OR use two-sided p from posterior mean/SE
* Combine z across guides via Stouffer weighted by inverse SE at best day.
* Then compute combined p and map to lfsr-like q using BH (documented approximation).
  (Implement only if quick; otherwise stick to min_lfsr.)

Apply optional effect size filter:

* require `max_d |postmean_gene_d| > effect_size_eps` if `effect_size_eps > 0`.

---

# 4) Update `Snakefile`

## Replace or extend rules

### 4.1 fit rule outputs two CSVs

`fit_pyro_export` produces:

* `{OUT}/gene_daywise_for_mash.csv`
* `{OUT}/guide_daywise_for_mash.csv`

### 4.2 mash rules (two modes) for genes and guides

For each mode in config mash_modes:

* `run_mash_gene_<mode>`: in gene csv -> mash_gene_<mode>.csv
* `run_mash_guide_<mode>`: in guide csv -> mash_guide_<mode>.csv

### 4.3 aggregate guides to genes per mode

* `aggregate_gene_<mode>`: mash_guide_<mode>.csv -> gene_from_guide_mash_<mode>.csv

### 4.4 choose default mode for final hits

* `rank_hits` uses `gene_from_guide_mash_{mash_default_mode}.csv` by default.

### 4.5 comparison rule

* `compare_modes`: consumes gene_from_guide_mash_<mode>.csv for all modes and writes mash_mode_comparison.csv

---

## Sherlock resources

* Pyro fit: GPU partition, `--gres=gpu:1`
* mashr + aggregation: normal partition CPU
  Keep current profile usage unchanged.

---

## Acceptance tests

1. Pipeline produces both exporter CSVs.
2. mashr runs in both modes and outputs four files:

   * mash_gene_conservative/enriched
   * mash_guide_conservative/enriched
3. Aggregator produces gene_from_guide_mash_<mode>.csv for each mode.
4. hits_ranked.csv produced using default mode from config.
5. Comparison table created with correct hit counts and overlap.

---

## Non-goals / DO NOT CHANGE

* Do not change the underlying Pyro likelihood or covariate structure.
* Do not add interaction terms between guides.
* Do not require new data or additional experiments.
* Do not remove NEU from the simplex model.
* Do not remove time smoothing; only change mashr settings and add guide-level EB.

END SPEC

```
::contentReference[oaicite:0]{index=0}
```
