**Methods-ready spec (copy/paste style)**

We model how CRISPRi guide content in high-MOI cells shifts CellRank fate probabilities across three terminal fates (EC, MES, NEU). For each cell (i), CellRank provides a simplex vector (p_i=(p_{i,\mathrm{EC}},p_{i,\mathrm{MES}},p_{i,\mathrm{NEU}})) with (\sum_f p_{i,f}=1). We define model-predicted fate probabilities (\pi_i=\mathrm{softmax}(\eta_i)), where EC is the reference fate ((\eta_{i,\mathrm{EC}}=0)) and (\eta_{i,f^\star}) are the logits for (f^\star\in{\mathrm{MES},\mathrm{NEU}}). The likelihood uses the “soft-label multinomial” / cross-entropy objective
[
\log L=\sum_{i=1}^N\sum_{f\in{\mathrm{EC,MES,NEU}}} p_{i,f}\log \pi_{i,f},
]
which corresponds to maximizing the expected categorical log-likelihood under fractional fate assignments (and is implemented stably via (\log\pi_i=\log\mathrm{softmax}(\eta_i))).

To handle MOI(\approx 5), each cell contains multiple guides; we therefore estimate guide effects jointly by summing the contributions of all detected guides in each cell. Let (d_i\in{0,1,2,3}) denote day, (r_i\in{0,1}) denote replicate, and (\tilde{k}*i) be the guide-count covariate (centered within day). Using an embedding-sum representation with padded guide lists (\text{guide_ids}*{i,m}) for (m=1..K_{\max}) and mask (\text{mask}*{i,m}), the linear predictor is
[
\eta*{i,f^\star}=\alpha_{f^\star,d_i}+b_{f^\star,r_i}+\gamma_{f^\star}\tilde{k}*i+\sum*{m=1}^{K_{\max}}\text{mask}*{i,m},\beta*{\text{guide_ids}*{i,m},f^\star,d_i}.
]
Guide-by-day effects decompose hierarchically into a gene-by-day effect plus a guide deviation:
[
\beta*{g,f^\star,d}=\theta_{\ell(g),f^\star,d}+\delta_{g,f^\star},
]
where (\ell(g)) maps each guide to its target gene, (\theta_{\ell,f^\star,d}) is the gene-level trajectory (reported effect), and (\delta_{g,f^\star}) captures guide-specific efficacy/off-target deviations shared across days. Gene effects are smoothed over time via a fate-specific random walk:
[
\theta_{\ell,f^\star,0}=\tau_{f^\star}z_{\ell,f^\star},\quad z_{\ell,f^\star}\sim\mathcal{N}(0,1),
]
[
\theta_{\ell,f^\star,d}=\theta_{\ell,f^\star,d-1}+\sigma_{\mathrm{time},f^\star}\varepsilon_{\ell,f^\star,d},\quad \varepsilon_{\ell,f^\star,d}\sim\mathcal{N}(0,1),\ d=1,2,3,
]
with weakly regularizing HalfNormal priors on (\tau_{f^\star}), (\sigma_{\mathrm{time},f^\star}), and (\sigma_{\mathrm{guide},f^\star}), and Gaussian priors for nuisance parameters (\alpha_{f^\star,d}), (b_{f^\star,r}), and (\gamma_{f^\star}). The model is fit with stochastic variational inference (SVI) using minibatches over cells.

We define the day-specific MES–EC gene effect as (\Delta_\ell(d)=\theta_{\ell,\mathrm{MES},d}) (since EC is the reference). To summarize per gene across days, we use a weighted average (\Delta_\ell^{\mathrm{sum}}=\sum_d w_d,\Delta_\ell(d)) (default (w_d\propto n_d), the number of retained cells at day (d)). Posterior samples from the variational approximation yield (\hat\Delta_\ell^{\mathrm{sum}}) (posterior mean) and an uncertainty estimate (\mathrm{se}*\ell) (posterior SD, optionally replaced by a small stratified bootstrap SE if VI appears overconfident). We then apply Empirical Bayes adaptive shrinkage to ((\hat\Delta*\ell^{\mathrm{sum}},\mathrm{se}_\ell)) across genes to estimate the across-gene effect distribution (g(\Delta)) and compute gene-level local false sign rates (lfsr) / q-values, following the adaptive-shrinkage framework used in TRADE. ([PubMed][1])

