````markdown
# CODEX SPEC v3.4: Identify θ vs δ by enforcing within-gene mean-zero δ (SVI-safe Pyro reparameterization)

## Goal
Fix the θ/δ non-identifiability in the hierarchical guide→gene model

\[
\beta_{g,d}=\theta_{\ell(g),d}+\delta_g
\]

by enforcing a **within-gene mean-zero constraint** on guide deviations:

\[
\sum_{g\in G_\ell} \delta_g = 0 \quad \forall\ \ell
\]
(or equivalently, mean(δ_g) = 0 within each gene).

This makes θ interpretable as **gene-average guide effect by day** (gene biology), preventing the per-gene constant shift that is currently causing daywise mis-localization (spurious θ on inactive days and depressed θ on true active day).

Scope: modify the Pyro model implementation only (plus small data-prep additions). Do not change mashr scripts, Snakemake, or evaluation logic in this spec.

---

## Background (why this is needed)
Currently, for each gene ℓ you have an invariance:
- θ_{ℓ,d} ← θ_{ℓ,d} − c_ℓ (for all d)
- δ_g ← δ_g + c_ℓ (for all g targeting gene ℓ)

leaving β unchanged. With broad priors, SVI can pick arbitrary c_ℓ and push mass into δ̄_ℓ, which corrupts θ-based day calling.

We remove this invariance by constraining δ within each gene.

---

## Model change (high-level)
Replace the unconstrained guide deviations

- δ_g = σ_guide * u_g,   u_g ~ Normal(0,1)

with a **gene-centered** deviation:

1) sample unconstrained u_g ~ Normal(0,1)
2) for each gene ℓ, compute mean(u_g) over guides in that gene
3) subtract that mean from u_g to get u'_g with mean zero within gene
4) define δ_g = σ_guide * u'_g

This guarantees within-gene mean-zero δ.

Important: This is a deterministic transform, SVI-friendly, and does not require a hard constraint distribution.

---

## Data requirements / precomputed mappings
You already have:
- G guides, L genes
- mapping ℓ(g) for each guide g

Add the following to the model inputs (tensors on device):

### Required tensors
1) `guide_to_gene: LongTensor[G]`
- guide_to_gene[g] = gene_id in [0..L-1]

2) `n_guides_per_gene: LongTensor[L]`
- counts of guides targeting each gene (>=1)

3) `gene_offsets: LongTensor[L+1]` (optional but recommended for fast grouping)
- CSR-style offsets for per-gene guide lists

4) `guides_by_gene: LongTensor[G]` (optional but recommended)
- concatenation of guide indices grouped by gene

You can build (3) and (4) once in preprocessing from `guide_to_gene`.

### Acceptance criteria for mappings
- Every guide belongs to exactly one gene
- Every gene has at least 1 guide (if not, either drop gene or handle separately)
- If a gene has 1 guide, centering makes u'_g = 0, so δ_g = 0 (this is desired; otherwise δ and θ are not separable for that gene)

---

## Pyro model changes (detailed)

### Where to apply
In the Pyro model file where you currently sample:
- `sigma_guide[f*] ~ HalfNormal(s_guide)`
- `u_g[f*] ~ Normal(0,1)` and set `delta_g[f*] = sigma_guide[f*] * u_g[f*]`

Modify that block to produce **centered δ**.

### Shape conventions
Let:
- Fstar = 2 (MES, NEU)  # EC is reference
- G = number of guides
- L = number of genes

We implement per-f* centering (same mapping across f*):
- u_raw: [Fstar, G]
- u_centered: [Fstar, G]
- delta: [Fstar, G]

### Centering transform (two implementation options)

#### Option A (recommended): scatter-add (fast, vectorized, no Python loops)
Inputs:
- `guide_to_gene` [G]

Steps:
1) Sample u_raw:
   - `u_raw ~ Normal(0,1)` with shape [Fstar, G]

2) Compute sum_u_per_gene per f*:
   - `sum_u = zeros([Fstar, L])`
   - `sum_u.scatter_add_(dim=1, index=guide_to_gene[None, :].expand(Fstar, G), src=u_raw)`

3) Compute mean_u_per_gene:
   - `mean_u = sum_u / n_guides_per_gene[None, :]`  # broadcast

4) Center per guide:
   - `u_centered = u_raw - mean_u.gather(dim=1, index=guide_to_gene[None,:].expand(Fstar,G))`

5) Define δ:
   - `delta = sigma_guide[:, None] * u_centered`

**Important Pyro constraint:** avoid in-place ops that break autograd graph provenance.
- Use functional ops (`torch.zeros(...).scatter_add` returning a new tensor) if your codebase has issues with `scatter_add_`.
- If using in-place, do it on a fresh tensor created in the same scope, not modifying something that is used elsewhere.

#### Option B: grouped indexing with offsets (safe, but may involve loops)
Use precomputed `gene_offsets` and `guides_by_gene`.
Loop over genes ℓ (L≈300 is fine):
- idx = guides_by_gene[offsets[ℓ]:offsets[ℓ+1]]
- mean_u = u_raw[:, idx].mean(dim=1, keepdim=True)
- u_centered[:, idx] = u_raw[:, idx] - mean_u
Then δ = σ * u_centered.

This is simpler to write correctly; L=300 makes it acceptable. Prefer Option A unless you’ve hit PyTorch scatter quirks.

---

## Implementation details: guide→gene θ block must remain unchanged
Keep your time random-walk prior on θ_{ℓ,f*,d}. No changes needed except that θ is now identified.

---

## Unit tests / sanity checks to add (must)
Add a small test function (or debug assertions under a flag) after computing delta:

1) Mean-zero check (numerical tolerance):
For each gene ℓ and each f*:
- `abs(delta[f*, guides_in_gene].mean()) < 1e-5` (or 1e-4 for float32)

Implement efficiently:
- compute `sum_delta_per_gene` via scatter_add and confirm near zero.

2) Invariance removal check (optional):
Run one tiny synthetic batch, compute β = θ + δ and verify that adding constant c to all guides of a gene cannot be represented without changing δ mean-zero property.

---

## Expected behavior changes (what to look for)
After refit on the same sim dataset:
- `gene_daywise_for_mash.csv` θ should align with truth θ day patterns
- The “constant offset across days” seen when comparing mean(β) - θ should disappear
- Daywise confusion should improve (especially day localization), while any-day should remain good
- Posterior δ will now represent **relative guide deviations within a gene**, not gene-level offsets

---

## CLI / preprocessing updates (minimal)
Where you build model inputs, add creation of:
- `guide_to_gene` tensor
- `n_guides_per_gene` tensor
Optionally:
- `gene_offsets`, `guides_by_gene`

These can be stored in your existing `.pt` cache or passed directly.

Pseudo-code for preprocessing:
```python
# guide_to_gene: array len G with gene_id per guide
guide_to_gene = np.array([gene_id_of_guide[g] for g in range(G)], dtype=np.int64)

n_guides_per_gene = np.bincount(guide_to_gene, minlength=L).astype(np.int64)

# optional grouped indices
order = np.argsort(guide_to_gene)
guides_by_gene = order.astype(np.int64)
gene_offsets = np.zeros(L+1, dtype=np.int64)
gene_offsets[1:] = np.cumsum(n_guides_per_gene)
````

---

## Acceptance criteria

1. Model runs SVI without graph/provenance errors (no in-place issues).
2. Debug check shows mean(δ_g | gene ℓ) ≈ 0 for all genes and both f*.
3. On sim evaluation, daywise confusion improves vs current v3.2 when using θ-based evaluation:

   * fewer daywise FPs (spurious inactive-day calls)
   * fewer daywise FNs on the true active day
4. `gene_daywise_for_mash.csv` remains in the same format (betahat_d*, se_d*), but values reflect identified θ.

---

## Notes / corner cases

* Genes with exactly 1 guide: δ_g becomes 0. This is correct; guide deviations are not identifiable there.
* If you later want a more flexible model, you can add a per-gene random effect in δ, but that would reintroduce shift unless constrained; keep this mean-zero approach as default.

```


