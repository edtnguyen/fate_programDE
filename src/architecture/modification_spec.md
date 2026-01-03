````markdown
# CODEX SPEC: Update Pyro model priors + switch discovery to mashr (Stanford Sherlock pipeline)

## Goal
Modify the existing fate-bias Pyro/SVI pipeline to:
1) keep the same model structure and data plumbing (CellRank soft-label likelihood, embedding-sum MOI handling, guide→gene hierarchy, burden/day/rep covariates),
2) make priors **least-assumptive** for a pre-screened 300-gene panel (avoid sparsity priors inside Pyro),
3) replace the “fixed time_scale c_d” with a **learned per-interval time innovation scale** (still smooth but not hard-coded to diverge later),
4) export **daywise MES–EC gene effects** and run **mashr across days** for hit calling (handles sign flips).

Keep all other unrelated parts intact.

---

## Files in repo (expected)
- `scripts/fit_pyro_export.py`  (Pyro model + SVI fit + export)
- `scripts/rank_hits.py`        (final ranking)
- `Snakefile`                   (workflow rules)
- `config.yaml`                 (paths + hyperparams)
- `envs/pyro.yaml`
- `envs/ash.yaml`               (may remain but no longer required for discovery)
- `scripts/run_ash.R`           (may remain but no longer used)
- `profiles/sherlock/config.yaml` (Slurm profile)

We will ADD:
- `envs/mash.yaml`
- `scripts/run_mashr.R`

---

## High-level model invariants (DO NOT CHANGE)
- Observed per-cell soft labels: `p_i = (p_EC, p_MES, p_NEU)` from `adata.obsm[fate_prob_key]`
- Likelihood remains soft-label multinomial cross-entropy:
  `sum_i sum_f p_{i,f} * log pi_{i,f}`, with `pi = softmax(eta)`
- EC is reference class: eta_EC = 0; learn logits for MES and NEU
- Embedding-sum MOI representation stays:
  `guide_ids: [N, Kmax]`, `mask: [N, Kmax]` (keep mask for NTC/padding disambiguation)
- Guide→gene hierarchy stays:
  `beta_{g,f*,d} = theta_{gene(g),f*,d} + delta_{g,f*}`
  with delta time-invariant
- Day and rep covariates stay:
  `alpha[f*, day] + b[f*, rep]`
- Burden term stays and remains centered within day:
  `gamma[f*] * k_centered`

---

## Required behavior changes

### Change A: Priors for alpha / rep / gamma become fixed-scale (remove hyperprior SDs)
Current priors likely sample `sigma_alpha`, `sigma_rep`, `sigma_gamma` as HalfNormals and then sample alpha/b/gamma conditional on them.

**Replace with fixed-scale weak Gaussian priors** (no hyper SD random variables):
- `alpha ~ Normal(0, s_alpha)` with shape [Fstar, D]
- `b ~ Normal(0, s_rep)` with shape [Fstar, R]
- `gamma ~ Normal(0, s_gamma)` with shape [Fstar]

Reason: D=4 and R=2; learning these SDs is weakly identified and slows SVI; fixed weak priors are less assumptive computationally and more stable.

### Change B: Day-0 global gene effect scale tau is kept, broad, and fate-specific
Keep:
- `tau[f*] ~ HalfNormal(s_tau)` (broad; no sparsity)
- `theta0[gene,f*] = tau[f*] * z0[gene,f*]`, `z0 ~ Normal(0,1)`

### Change C: Time smoothing changes from fixed `time_scale c_d` to learned per-interval innovation scales
Current implementation:
- `sigma_time[f*] ~ HalfNormal(s_time)`
- random walk increments multiplied by `c_d * sigma_time[f*]`

**Replace with per-interval scales**:
- `sigma_time[f*, d] ~ HalfNormal(s_time)` for d=0..D-2 (shape [Fstar, D-1])
- `theta_{d} = theta_{d-1} + sigma_time[:,d-1] * eps_{d-1}` with `eps ~ Normal(0,1)`

This preserves smoothness but does not hard-code larger divergence later.

### Change D: Keep guide deviation scale broad
Keep `sigma_guide[f*] ~ HalfNormal(s_guide)` and ensure s_guide is broad (e.g. 1–2). Do not force small.

### Change E: Add optional evidence weight for soft likelihood (default 1.0)
Add config parameter `likelihood_weight` (lambda). In likelihood term multiply:
- `pyro.factor("soft_ce", scale * lambda * ll)`

Default `lambda=1.0`. Do not change behavior unless user sets it.

### Change F: Export daywise MES–EC gene effects for mashr
Stop exporting a single across-day summary for discovery. Instead export:
- for each gene: `betahat_d0..d{D-1}` and `se_d0..se_d{D-1}` for MES logit only

Daywise estimates come from VI posterior draws:
- `betahat_d = posterior_mean(theta_mes[:, gene, d])`
- `se_d = posterior_sd(theta_mes[:, gene, d])`

### Change G: Switch Snakemake discovery path from ashr to mashr
- Add `envs/mash.yaml`
- Add `scripts/run_mashr.R` that consumes `gene_summary_for_mash.csv` and outputs `gene_summary_mash_out.csv`
- Update `Snakefile` to run mashr rule instead of ashr (ash can remain optional but should not be required)
- Update `scripts/rank_hits.py` to read mash output, rank by `lfsr_min` and `hit_anyday`

---

## Config changes (`config.yaml`)
Add these keys (with defaults):
- `s_alpha: 2.5`          # day intercept prior sd
- `s_rep: 0.8`            # rep effect prior sd
- `s_gamma: 0.8`          # burden effect prior sd
- `s_tau: 4.0`            # day0 gene scale HalfNormal
- `s_time: 0.8`           # per-interval time scale HalfNormal
- `s_guide: 1.5`          # guide deviation scale HalfNormal
- `likelihood_weight: 1.0`
- `mash_lfsr_thresh: 0.05`

Do not remove existing keys. Keep all old keys compatible.

---

## Detailed code edits

### 1) `scripts/fit_pyro_export.py`

#### 1.1) In `fate_model(...)` signature
Add params (passed or read from cfg):
- `s_alpha, s_rep, s_gamma, s_tau, s_time, s_guide, likelihood_weight`

Keep existing args and logic.

#### 1.2) Replace alpha/b/gamma priors block
**Before (example)**:
- sampling `sigma_alpha`, then `alpha` conditioned on it

**After** (fixed):
- `alpha = pyro.sample("alpha", Normal(0, s_alpha).expand([Fstar, D]).to_event(2))`
- `b = pyro.sample("b", Normal(0, s_rep).expand([Fstar, R]).to_event(2))`
- `gamma = pyro.sample("gamma", Normal(0, s_gamma).expand([Fstar]).to_event(1))`

#### 1.3) Replace time smoothing block
**Keep non-centered RW sampling of z0 and eps**, but change sigma_time shape:

- `tau = pyro.sample("tau", HalfNormal(s_tau).expand([Fstar]).to_event(1))`
- `z0 = pyro.sample("z0", Normal(0,1).expand([L, Fstar]).to_event(2))`
- `theta0 = tau.unsqueeze(0) * z0`   # [L,Fstar]

**Old**:
- `sigma_time ~ HalfNormal(s_time).expand([Fstar])`
- increments multiplied by `c_d`

**New**:
- `sigma_time = pyro.sample("sigma_time", HalfNormal(s_time).expand([Fstar, D-1]).to_event(2))`
- `eps = pyro.sample("eps", Normal(0,1).expand([L, Fstar, D-1]).to_event(3))`
- `inc = eps * sigma_time.unsqueeze(0)`  # broadcasting -> [L,Fstar,D-1]
- `theta[:,:,1:] = theta0.unsqueeze(-1) + torch.cumsum(inc, dim=-1)`
- `theta[:,:,0] = theta0`

Then prepend gene-0 row of zeros as before.

#### 1.4) Guide deviation block: keep but ensure broad s_guide
- `sigma_guide ~ HalfNormal(s_guide).expand([Fstar])`
- `delta = sigma_guide.unsqueeze(0) * u`
- prepend guide-0 row zeros as before.

#### 1.5) Likelihood term scaling
Existing:
- `scale = N/B`
- `pyro.factor("soft_ce", scale * ll)`

New:
- `pyro.factor("soft_ce", scale * likelihood_weight * ll)`

Do NOT detach tensors used in `ll`.
Do NOT use in-place logits construction; keep `eta = torch.cat([zeros, eta_star], dim=1)` style.

#### 1.6) Update `reconstruct_theta_samples(...)`
Because `sigma_time` is now [S,Fstar,D-1], adjust reconstruction:

- `tau: [S,Fstar]`
- `z0: [S,L,Fstar]`
- `sigma_time: [S,Fstar,D-1]`
- `eps: [S,L,Fstar,D-1]`

Compute:
- `theta0 = tau[:,None,:] * z0`
- `inc = eps * sigma_time[:,None,:,:]`  # -> [S,L,Fstar,D-1]
- `theta[...,0] = theta0`
- `theta[...,1:] = theta0[...,None] + cumsum(inc, -1)`

Return [S,L,Fstar,D].

#### 1.7) Export function for mashr
Add or replace exporter so output CSV contains:

Columns:
- `gene`
- `betahat_d0 ... betahat_d3` (MES dimension only)
- `se_d0 ... se_d3`

Use posterior draws from guide:
- `theta = reconstruct_theta_samples(...)`
- `theta_mes = theta[:,:,MES_INDEX,:]` where MES_INDEX = 0 in `Fstar` ordering (eta_star[:,0] corresponds to MES)
- betahat = mean over draws; se = sd over draws

**Ensure** this exporter writes to the path requested by CLI `--out`.

Do NOT remove older exporter functions; you may leave them but pipeline should call mash exporter.

#### 1.8) Wire cfg hyperparams into `fate_model`
When calling `fate_model(...)` inside `_model(...)`, pass:
- s_alpha, s_rep, s_gamma, s_tau, s_time, s_guide, likelihood_weight from cfg with defaults.

---

### 2) Add `envs/mash.yaml`
Create the file exactly:

```yaml
name: fate-mash
channels:
  - conda-forge
dependencies:
  - r-base>=4.3
  - r-data.table
  - r-ashr
  - r-mashr
````

---

### 3) Add `scripts/run_mashr.R`

Input: `gene_summary_for_mash.csv`
Output: `gene_summary_mash_out.csv`

Expected input columns: `gene`, `betahat_d0..d3`, `se_d0..d3`

Behavior:

* Build `Bhat` = matrix L x D from betahat columns
* Build `Shat` = matrix L x D from se columns
* `data <- mash_set_data(Bhat, Shat)`
* `Vhat <- estimate_null_correlation_simple(data)`
* `data <- mash_set_data(Bhat, Shat, V=Vhat)`
* `U.c <- cov_canonical(data)`
* `m <- mash(data, U.c)`
* Get `postmean <- get_pm(m)` and `lfsr <- get_lfsr(m)`
* Add columns:

  * `postmean_d0..d3`
  * `lfsr_d0..d3`
  * `lfsr_min = pmin(lfsr_d0..d3)`
  * `hit_anyday = (lfsr_min < mash_lfsr_thresh)` where threshold is applied later in Python (R may hardcode 0.05 or write lfsr_min only)
  * `best_day` = day with smallest lfsr (argmin)

Write output CSV.

---

### 4) Update `Snakefile`

Replace/extend the workflow so that:

* `fit_pyro_export` outputs: `{OUT}/gene_summary_for_mash.csv`
* add rule `run_mashr` consuming that and producing `{OUT}/gene_summary_mash_out.csv` using `envs/mash.yaml`
* `rank_hits` consumes mash output, produces `{OUT}/hits_ranked.csv`

Keep Sherock resource fields exactly as currently used (partition/gres/mem/time strings).

Do not delete existing `run_ash` rule if present; it can remain unused.

---

### 5) Update `scripts/rank_hits.py`

Change CLI to accept:

* `--mash` instead of `--ash` (keep backward compatibility if easy, but not required)

Ranking behavior:

* read mash output CSV
* compute `lfsr_min` if not present
* call hit if `lfsr_min < config["mash_lfsr_thresh"]` (default 0.05)
* sort by `hit_anyday desc`, then `lfsr_min asc`
* write `hits_ranked.csv`

---

## Acceptance tests (must pass)

1. `snakemake --profile profiles/sherlock -j <N>` completes end-to-end and produces:

   * `out_dir/gene_summary_for_mash.csv`
   * `out_dir/gene_summary_mash_out.csv`
   * `out_dir/hits_ranked.csv`

2. `gene_summary_for_mash.csv` has exactly:

   * columns: `gene`, betahat_d0..d3, se_d0..d3
   * row count equals #genes (300) excluding baseline gene 0

3. `gene_summary_mash_out.csv` has:

   * lfsr_d0..d3, postmean_d0..d3, lfsr_min, best_day

4. Model still:

   * uses embedding-sum per batch (no sparse mm required)
   * keeps burden centered within day
   * keeps guide deviations time-invariant
   * uses `torch.cat([zeros, eta_star], dim=1)` (no in-place writes)

---

## Notes / non-goals (DO NOT DO)

* Do not introduce horseshoe / spike-and-slab / point-mass sparsity priors in Pyro.
* Do not remove NEU logit from the model; keep simplex (EC/MES/NEU).
* Do not change data extraction keys (`adata.obs["day"]`, `adata.obsm["covar"]["rep"]`, `adata.obsm["guide"]`).
* Do not change guide→gene mapping behavior (NTC hard-zero baseline).
* Do not change training loop (SVI) other than passing new hyperparams.

---

## Optional (only if trivial)

* Keep old ash path as an optional rule `run_ash` but not part of `rule all`.
* Add a config switch `discovery_method: mashr|ashr` (default mashr). Only implement if very easy.

END SPEC

```
::contentReference[oaicite:0]{index=0}
```
