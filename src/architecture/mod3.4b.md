```markdown
# ADD-ON SPEC v3.4b: Ensure exported gene_daywise_for_mash.csv is truly θ (not β / not offset) + export diagnostics

## Goal
After implementing v3.4 (within-gene mean-zero δ), make the **export layer** unambiguous and verifiable:
1) `gene_daywise_for_mash.csv` must contain **θ̂(gene, day)** (MES–EC) with SE from the Pyro posterior.
2) `guide_daywise_for_mash.csv` must contain **β̂(guide, day) = θ̂(gene(guide), day) + δ̂(guide)** with SE.
3) Add a diagnostic export that confirms the constraint and catches regressions:
   - `qc_theta_beta_offset_by_gene.csv`
   - `qc_delta_mean_by_gene.csv`

Scope: modify only the Python export code (posterior summary code path), not the Pyro model beyond v3.4.

---

## 1) Export contract (exact meanings)

### 1.1 gene_daywise_for_mash.csv (θ only)
Columns (unchanged naming):
- `gene`
- `betahat_d0..betahat_d{D-1}` = posterior mean of θ_{gene,MES,day}
- `se_d0..se_d{D-1}` = posterior SD (or SE proxy) of θ_{gene,MES,day}

**Must not** include δ, must not be derived from guide aggregation.

### 1.2 guide_daywise_for_mash.csv (β only)
Columns (example; keep your existing column names but ensure content):
- `guide`
- `gene` (optional but helpful)
- `betahat_d*` = posterior mean of β_{guide,MES,day} = θ_{gene(guide),day} + δ_{guide}
- `se_d*` = posterior SD of β_{guide,MES,day}

### 1.3 Separate θ and δ exports (NEW, for debugging)
Add:
- `theta_posterior_summary.npz` (or .pt) containing arrays:
  - `theta_mean` shape [L, D] for MES (and optionally [F*, L, D] if both f*)
  - `theta_sd`   shape [L, D]
- `delta_posterior_summary.npz` containing:
  - `delta_mean` shape [G] (or [F*, G])
  - `delta_sd`   shape [G] (or [F*, G])

These are intermediate artifacts used to compute CSVs; they simplify debugging.

---

## 2) Implementation: posterior summary extraction

### 2.1 Posterior draws source
Use your existing VI posterior sampling mechanism:
- `guide = pyro.infer.autoguide.*` already trained
- draw S samples (e.g., S=200–1000) from guide posterior
- compute sample means and sample stds

**Do not** reuse mash outputs to define these; the export is *pre-mash*.

### 2.2 Exact tensors to extract
You must extract:
- θ_{MES}: `theta_samples` shape [S, L, D]
- δ_{MES}: `delta_samples` shape [S, G]  (after centering transform! i.e., constrained δ)

Then:
- `theta_mean = theta_samples.mean(0)` → [L, D]
- `theta_sd   = theta_samples.std(0, unbiased=False)` → [L, D]
- `delta_mean = delta_samples.mean(0)` → [G]
- `delta_sd   = delta_samples.std(0, unbiased=False)` → [G]

### 2.3 Compute β samples for guide export
For each sample s:
- `beta_samples[s, g, d] = theta_samples[s, gene(g), d] + delta_samples[s, g]`
Then:
- `beta_mean = beta_samples.mean(0)` → [G, D]
- `beta_sd   = beta_samples.std(0, unbiased=False)` → [G, D]

Note: β SD is not just sqrt(theta_sd^2 + delta_sd^2) because θ and δ can be correlated in posterior; compute SD from samples.

---

## 3) Write CSVs (exact formatting)

### 3.1 gene_daywise_for_mash.csv
Create dataframe rows for each gene ℓ:
- `gene` = gene_name
- `betahat_d{d}` = theta_mean[ℓ, d]
- `se_d{d}`      = theta_sd[ℓ, d]

Write to:
- `OUT/gene_daywise_for_mash.csv`

### 3.2 guide_daywise_for_mash.csv
Rows for each guide g:
- `guide` = guide_name
- `gene` = gene_name_of_guide
- `betahat_d{d}` = beta_mean[g, d]
- `se_d{d}`      = beta_sd[g, d]

Write to:
- `OUT/guide_daywise_for_mash.csv`

---

## 4) NEW diagnostics exports (must)

### 4.1 qc_delta_mean_by_gene.csv (checks the constraint in posterior mean)
Compute per gene ℓ:
- `delta_mean_gene = mean_{g in Gℓ} delta_mean[g]`
- `delta_sd_gene`  = std_{g in Gℓ} delta_mean[g]` (spread across guides)

Write:
- `gene, n_guides, delta_mean_gene, delta_sd_gene`
Expected:
- `delta_mean_gene` near 0 for all genes (tolerance ~1e-3 to 1e-2 depending on posterior noise)

### 4.2 qc_theta_beta_offset_by_gene.csv (detects “constant offset across days” regressions)
Compute, for each gene ℓ and day d:
- `beta_mean_gene_day = mean_{g in Gℓ} beta_mean[g, d]`
- `offset_day = beta_mean_gene_day - theta_mean[ℓ, d]`

Then summarize per gene:
- `offset_mean = mean_d offset_day`
- `offset_sd_across_days = std_d offset_day`

Write:
- `gene, n_guides, offset_mean, offset_sd_across_days`
Expected after v3.4:
- `offset_mean` ~ 0
- `offset_sd_across_days` ~ 0
If either is large, export is wrong or δ-centering is not applied.

### 4.3 Optional: qc_theta_day_centering.csv (helps day localization)
Compute per gene:
- `theta_mean_centered = theta_mean[ℓ,:] - theta_mean[ℓ,:].mean()`
Write a quick summary:
- `gene, theta_mean_d0..d3, theta_centered_d0..d3`

This is optional; useful if later you want to evaluate purely “shape across days”.

---

## 5) Safety checks and failure modes
Add hard assertions (or warnings that fail CI in tests) in export code:

1) Check centered δ posterior mean per gene:
- `max_abs(delta_mean_gene) < 0.02`  (tune threshold; start 0.02)

2) Check offset regression:
- `max_abs(offset_mean) < 0.02`
- `max(offset_sd_across_days) < 0.02`

If violated:
- print top-10 genes with largest violations
- raise RuntimeError in simulation mode; warn in real mode

---

## 6) Acceptance criteria
After implementing v3.4 + this add-on:
1) `gene_daywise_for_mash.csv` matches θ truth patterns in sim (day-local).
2) `qc_delta_mean_by_gene.csv` shows near-zero means for all genes.
3) `qc_theta_beta_offset_by_gene.csv` shows near-zero offsets; no “constant across day” offsets remain.
4) Guide export still matches previous behavior (β), but now β’s gene-mean matches θ.

---

## 7) Implementation checklist
1) Locate current export code path that generates `gene_daywise_for_mash.csv` and `guide_daywise_for_mash.csv`.
2) Ensure it is using posterior draws and writing θ and β as specified.
3) Add new `.npz` (or `.pt`) intermediate saves for θ and δ summaries.
4) Add two QC CSVs and assertions.
5) Run on sim dataset and confirm QC passes; then re-run mash and evaluation.

```

