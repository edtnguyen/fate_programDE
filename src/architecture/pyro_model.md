1. Observed data (after filtering)

* Cells (i=1,\dots,N) after filtering to (k_i \le K_{\max}) with (K_{\max}=20), where (k_i) is the number of detected guides in cell (i).
* Days (d_i \in {0,1,2,3}) (d0–d3).
* Replicates (r_i \in {0,1}).
* CellRank fate probabilities (given):
  [
  p_i = (p_{i,\mathrm{EC}},p_{i,\mathrm{MES}},p_{i,\mathrm{NEU}}),\qquad \sum_f p_{i,f}=1.
  ]
* Guides: total non-NTC guides (g \in {1,\dots,G}) with (G=2000).
* Genes: targets (\ell \in {1,\dots,L}) with (L=300).
* Guide→gene map (\ell(g)\in{1,\dots,L}).

Embedding-sum representation (per cell):

* `guide_ids[i,m]` for (m=1,\dots,K_{\max}), integer in ({0,\dots,G}).
* `mask[i,m]\in\{0,1\}` indicates real vs padding.

Hard-zero convention:

* Map all NTC guides to `guide_id = 0` (real entry, so `mask=1`).
* Padding positions also use `guide_id=0` but with `mask=0`.
* Define `gene_of_guide[0]=0` and `gene_of_guide[g]=\ell(g)` for (g\ge1).

Define centered guide burden covariate (recommended):
[
\tilde{k}*i = k_i - \bar{k}*{d_i}\quad \text{(center within day)}.
]

---

2. Latent parameters and indexing

Fates:

* Reference fate: EC.
* Non-reference fates: (\mathcal{F}^\star = {\mathrm{MES},\mathrm{NEU}}).
  Index them as (f^\star \in {1,2}) corresponding to (MES, NEU).

Core latent effects:

* Gene-by-day effects: (\theta_{\ell,f^\star,d}) for (\ell=1..L), (f^\star\in\mathcal{F}^\star), (d=0..3).
* Guide deviations (time-invariant): (\delta_{g,f^\star}) for (g=1..G), (f^\star\in\mathcal{F}^\star).

Nuisance:

* Day intercepts: (\alpha_{f^\star,d}).
* Replicate random effects: (b_{f^\star,r}).
* Burden slope: (\gamma_{f^\star}).

Hard-zero baseline rows:
[
\theta_{0,f^\star,d}\equiv 0,\qquad \delta_{0,f^\star}\equiv 0.
]

---

3. Guide-level effect decomposition (handles multiple guides/gene)

For any non-NTC guide (g\ge1):
[
\beta_{g,f^\star,d} = \theta_{\ell(g),f^\star,d} + \delta_{g,f^\star}.
]
For (g=0) (NTC/baseline):
[
\beta_{0,f^\star,d}\equiv 0.
]

Interpretation:

* (\theta_{\ell,f^\star,d}): gene-level effect trajectory (what you ultimately report).
* (\delta_{g,f^\star}): guide-specific deviation (dud/off-target), shared across days.

---

4. Cell-level linear predictor (MOI≈5 handled by summing all guides present)

For each cell (i) and non-reference fate (f^\star\in{\mathrm{MES},\mathrm{NEU}}):
[
\eta_{i,f^\star}
================

\alpha_{f^\star,d_i}
+
b_{f^\star,r_i}
+
\gamma_{f^\star}\tilde{k}*i
+
\sum*{g\in G_i}\beta_{g,f^\star,d_i},
\qquad \eta_{i,\mathrm{EC}}=0.
]

Embedding-sum form using padded lists:
[
\sum_{g\in G_i}\beta_{g,f^\star,d_i}
====================================

\sum_{m=1}^{K_{\max}}
\text{mask}*{i,m};
\beta*{\text{guide_ids}_{i,m},f^\star,d_i}.
]

This is the “MOI fix”: each guide’s effect is estimated conditional on other guides co-occurring in the same cell.

---

5. Softmax mapping to predicted fate probabilities

Let (\eta_i = (\eta_{i,\mathrm{EC}},\eta_{i,\mathrm{MES}},\eta_{i,\mathrm{NEU}})) with (\eta_{i,\mathrm{EC}}=0). Then:
[
\pi_i = \mathrm{softmax}(\eta_i),\qquad
\pi_{i,f}=\frac{\exp(\eta_{i,f})}{\sum_{f'\in{\mathrm{EC,MES,NEU}}}\exp(\eta_{i,f'})}.
]

---

6. Likelihood (soft-label multinomial / cross-entropy with CellRank probabilities)

Use CellRank probabilities (p_i) as fractional labels:
[
\log p({p_i}\mid {\pi_i})
=========================

\sum_{i=1}^{N}\sum_{f\in{\mathrm{EC,MES,NEU}}} p_{i,f}\log \pi_{i,f}.
]

Implementation uses (\log\pi_i=\log\mathrm{softmax}(\eta_i)) via `log_softmax`.

---

7. Priors (weakly regularizing; not “most genes are null”)

7.1 Day intercepts (fate-specific):
[
\alpha_{f^\star,d}\sim \mathcal{N}(0,\sigma_{\alpha,f^\star}^2),\qquad
\sigma_{\alpha,f^\star}\sim \mathrm{HalfNormal}(s_\alpha).
]

7.2 Replicate random effects:
[
b_{f^\star,r}\sim \mathcal{N}(0,\sigma_{\mathrm{rep},f^\star}^2),\qquad
\sigma_{\mathrm{rep},f^\star}\sim \mathrm{HalfNormal}(s_{\mathrm{rep}}).
]

7.3 Burden slope:
[
\gamma_{f^\star}\sim \mathcal{N}(0,\sigma_{\gamma,f^\star}^2),\qquad
\sigma_{\gamma,f^\star}\sim \mathrm{HalfNormal}(s_\gamma).
]

7.4 Gene-by-day effects with smooth time evolution (random walk)

Fate-specific global scale for day 0:
[
\tau_{f^\star}\sim \mathrm{HalfNormal}(s_\tau).
]
Day 0:
[
\theta_{\ell,f^\star,0} = \tau_{f^\star} z_{\ell,f^\star,0},\qquad
z_{\ell,f^\star,0}\sim \mathcal{N}(0,1).
]

Time smoothing:
[
\sigma_{\mathrm{time},f^\star}\sim \mathrm{HalfNormal}(s_{\mathrm{time}}),
]
[
\theta_{\ell,f^\star,d} = \theta_{\ell,f^\star,d-1} + \sigma_{\mathrm{time},f^\star},\varepsilon_{\ell,f^\star,d},
\qquad
\varepsilon_{\ell,f^\star,d}\sim \mathcal{N}(0,1),\ d=1,2,3.
]

7.5 Guide deviations (time-invariant)
[
\sigma_{\mathrm{guide},f^\star}\sim \mathrm{HalfNormal}(s_{\mathrm{guide}}),
]
[
\delta_{g,f^\star} = \sigma_{\mathrm{guide},f^\star} u_{g,f^\star},\qquad
u_{g,f^\star}\sim \mathcal{N}(0,1).
]

Hard zero:
[
\theta_{0,f^\star,d}=0,\quad \delta_{0,f^\star}=0.
]

---

8. Non-centered parameterization (explicit)

You fit the non-centered latents ((z,\varepsilon,u)) and scales ((\tau,\sigma_{\mathrm{time}},\sigma_{\mathrm{guide}})), then deterministically build (\theta,\delta) as above. (This is what improves SVI stability.)

---

9. SVI training objective with minibatching

For a minibatch (S) of size (B):
[
\log p({p_i}\mid \phi)\approx \frac{N}{B}\sum_{i\in S}\sum_f p_{i,f}\log \pi_{i,f}(\phi).
]

Implementation:

* `pyro.plate("cells", N, subsample_size=B)` gives indices (S).
* Multiply minibatch log-likelihood by (N/B).
* Use `ClippedAdam` for stability.

---

10. Primary contrast: MES–EC

Because EC is the reference, the gene-level MES–EC log-odds effect at day (d) is:
[
\Delta_\ell(d) \equiv \theta_{\ell,\mathrm{MES},d}.
]

---

11. Gene summary across days (single number per gene)

Choose weights (w_d\ge 0), (\sum_d w_d=1). Recommended default:

* (w_d \propto n_d), number of retained cells at day (d).

Define:
[
\Delta^{\mathrm{sum}}*\ell = \sum*{d=0}^3 w_d,\Delta_\ell(d)
= \sum_{d=0}^3 w_d,\theta_{\ell,\mathrm{MES},d}.
]

From posterior draws of (\theta) (via the VI guide), compute:

* (\widehat{\Delta}^{\mathrm{sum}}*\ell) = posterior mean of (\Delta^{\mathrm{sum}}*\ell)
* (\mathrm{se}*\ell) = posterior sd of (\Delta^{\mathrm{sum}}*\ell)

(If you suspect VI underestimates uncertainty, replace (\mathrm{se}*\ell) with a small stratified subset bootstrap SE; keep (\widehat{\Delta}^{\mathrm{sum}}*\ell) the same.)

---

12. Empirical Bayes shrinkage + hit calling using ash (TRADE-style)

Observation model for gene summaries:
[
\widehat{\Delta}*\ell \mid \Delta*\ell \approx \mathcal{N}(\Delta_\ell,\ \mathrm{se}*\ell^2),
]
where (\widehat{\Delta}*\ell \equiv \widehat{\Delta}^{\mathrm{sum}}_\ell).

Across genes:
[
\Delta_\ell \sim g(\cdot),
]
with (g) estimated nonparametrically as a mixture:
[
g(\Delta)=\sum_k \pi_k f_k(\Delta)
]
(e.g., half-uniform components allowing asymmetric effects).

Recommended ash settings for your *enriched 300-hit set*:

* `pointmass = FALSE` (do not force a spike at 0)
* `mixcompdist = "halfuniform"` (flexible EB prior)
* use ash outputs:

  * posterior mean (shrunk effect)
  * `lfsr` (local false sign rate)
  * `qvalue` (FDR-style)

Hit calling:

* primary: low `lfsr` (e.g., < 0.05), optionally plus a minimum effect size on the shrunk posterior mean or posterior tail probability.

---

13. Minimal diagnostics (recommended)

* Compare held-out cross-entropy (test set) for:

  * nuisance-only model (no guide term)
  * full model (with guide term)
* Negative control: permute guides within (day,rep,k-bin) and confirm ash hit rates collapse.
* Sanity: known regulators show expected MES–EC direction.

