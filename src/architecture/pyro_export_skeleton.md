Yes — keeping the R step as a separate script and chaining with Snakemake is the cleanest / least finicky setup.

Below is a **full pipeline skeleton** that matches your AnnData layout:

* `day` from `adata.obs["day"]`
* `rep` from `adata.obsm["covar"]["rep"]`
* guide matrix from `adata.obsm["guide"]` (nonzero = guide present; if counts exist, we’ll use them to pick top-Kmax)

It’s split into:

* `Snakefile`
* `config.yaml`
* `scripts/fit_pyro_export.py`
* `scripts/run_ash.R`
* `scripts/rank_hits.py`
* optional `envs/*.yaml` if you want Snakemake-managed envs

---

## 1) `config.yaml`

```yaml
adata_path: data/adata.h5ad

# Where are CellRank fate probabilities stored?
# Examples you might use:
#   "lineages_fwd" (common in CellRank)
#   "fate_probs" (if you stored your own)
fate_prob_key: lineages_fwd

# Fate order in that matrix / dataframe:
# Must match the columns in adata.obsm[fate_prob_key]
fates: ["EC", "MES", "NEU"]

# Guide matrix key
guide_key: guide
covar_key: covar
rep_key: rep
day_key: day

# Guide-to-gene mapping file (must align with guide columns)
# Required columns:
#   guide_name, gene_name, is_ntc (0/1)
guide_map_csv: data/guide_map.csv

# Output dir
out_dir: out_fate_pipeline

# Model sizes
Kmax: 20
D: 4
R: 2

# SVI
batch_size: 8192
lr: 1e-3
clip_norm: 5.0
num_steps: 5000
num_posterior_draws: 1000
seed: 0

# Across-day summary weights:
# If null -> weights proportional to day cell counts after filtering
weights: null

# ashr thresholds for final "hit" label
lfsr_thresh: 0.05
qvalue_thresh: 0.10
```

---

## 2) `Snakefile`

```python
configfile: "config.yaml"

OUT = config["out_dir"]

rule all:
    input:
        f"{OUT}/hits_ranked.csv"

rule fit_pyro_export:
    input:
        adata=config["adata_path"],
        guide_map=config["guide_map_csv"]
    output:
        summary=f"{OUT}/gene_summary_for_ash.csv"
    params:
        cfg="config.yaml"
    threads: 4
    resources:
        gpu=1
    shell:
        r"""
        mkdir -p {OUT}
        python scripts/fit_pyro_export.py \
            --config {params.cfg} \
            --adata {input.adata} \
            --guide-map {input.guide_map} \
            --out {output.summary}
        """

rule run_ash:
    input:
        summary=f"{OUT}/gene_summary_for_ash.csv"
    output:
        ash=f"{OUT}/gene_summary_ash_out.csv"
    shell:
        r"""
        Rscript scripts/run_ash.R {input.summary} {output.ash}
        """

rule rank_hits:
    input:
        ash=f"{OUT}/gene_summary_ash_out.csv"
    output:
        hits=f"{OUT}/hits_ranked.csv"
    params:
        cfg="config.yaml"
    shell:
        r"""
        python scripts/rank_hits.py \
            --config {params.cfg} \
            --ash {input.ash} \
            --out {output.hits}
        """
```

---

## 3) `scripts/fit_pyro_export.py`

```python
#!/usr/bin/env python
import argparse, yaml
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import anndata as ad

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam


# ---------- utilities: parsing day/rep, extracting fate probs ----------

def parse_day_to_int(day_series: pd.Series, D: int) -> np.ndarray:
    """
    Converts adata.obs["day"] to integer 0..D-1.
    Handles:
      - already ints {0,1,2,3}
      - strings like "d0","d1","d2","d3"
      - strings like "0","1",...
    """
    vals = day_series.astype(str).values
    # try extracting digits
    day_num = np.array([int("".join([c for c in v if c.isdigit()])) for v in vals], dtype=np.int64)
    uniq = np.sort(np.unique(day_num))
    if len(uniq) != D:
        raise ValueError(f"Expected {D} unique days, got {uniq}")
    mapping = {u:i for i,u in enumerate(uniq)}
    return np.array([mapping[x] for x in day_num], dtype=np.int64)


def parse_rep_to_int(rep_array, R: int) -> np.ndarray:
    """
    Converts rep to integer 0..R-1.
    rep_array may be:
      - pandas Series
      - numpy array
    """
    rep = np.asarray(rep_array).astype(str)
    uniq = np.sort(np.unique(rep))
    if len(uniq) != R:
        raise ValueError(f"Expected {R} unique reps, got {uniq}")
    mapping = {u:i for i,u in enumerate(uniq)}
    return np.array([mapping[x] for x in rep], dtype=np.int64)


def get_fate_probs(adata, key: str, fates: list) -> np.ndarray:
    """
    Returns p: [N,3] float32 in order fates (e.g., EC, MES, NEU).
    Supports:
      - adata.obsm[key] as pandas DataFrame with columns containing fate names
      - adata.obsm[key] as numpy array (then assumes already ordered correctly)
    """
    obj = adata.obsm[key]
    if hasattr(obj, "columns"):  # DataFrame
        missing = [f for f in fates if f not in obj.columns]
        if missing:
            raise ValueError(f"Missing fates {missing} in adata.obsm['{key}'].columns={list(obj.columns)}")
        p = obj[fates].to_numpy(dtype=np.float32)
    else:
        p = np.asarray(obj, dtype=np.float32)
        if p.shape[1] != len(fates):
            raise ValueError(f"adata.obsm['{key}'] has shape {p.shape}, expected second dim={len(fates)}")
    # normalize defensively
    p = np.clip(p, 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


# ---------- guide matrix -> padded guide_ids/mask with hard-zero NTC ----------

def load_guide_map(path: str) -> pd.DataFrame:
    gm = pd.read_csv(path)
    required = {"guide_name", "gene_name", "is_ntc"}
    if not required.issubset(set(gm.columns)):
        raise ValueError(f"guide_map_csv must contain columns {required}, got {gm.columns}")
    return gm


def get_guide_columns(guide_obsm):
    """
    guide_obsm may be:
      - pandas DataFrame (preferred): columns are guide names
      - scipy sparse / numpy array: then we require adata.uns['guide_names'] in the caller
    """
    if hasattr(guide_obsm, "columns"):
        return list(guide_obsm.columns), guide_obsm.to_numpy()
    else:
        return None, guide_obsm


def build_id_maps(guide_names, guide_map_df):
    """
    Map:
      - guide_name -> guide_id in {0..G}, where 0 is NTC hard-zero
      - guide_id -> gene_id in {0..L}, where 0 is baseline hard-zero
      - gene_id -> gene_name (1..L)
    """
    # genes (exclude NTC rows when building gene list)
    gene_names = sorted(set(guide_map_df.loc[guide_map_df["is_ntc"] == 0, "gene_name"].tolist()))
    gene_to_id = {g:i+1 for i,g in enumerate(gene_names)}  # 1..L
    L = len(gene_names)

    # assign guide_id: 0 for all NTC; 1.. for non-NTC (stable order by guide_name)
    nonntc_guides = sorted(guide_map_df.loc[guide_map_df["is_ntc"] == 0, "guide_name"].tolist())
    guide_to_id = {g:i+1 for i,g in enumerate(nonntc_guides)}  # 1..G
    guide_to_id_ntc = {}  # all NTC -> 0
    for g in guide_map_df.loc[guide_map_df["is_ntc"] == 1, "guide_name"].tolist():
        guide_to_id_ntc[g] = 0

    # final guide_name -> guide_id
    guide_name_to_gid = {}
    guide_name_to_gid.update(guide_to_id)
    guide_name_to_gid.update(guide_to_id_ntc)

    G = len(nonntc_guides)

    # guide_id -> gene_id (length G+1)
    gid_to_gene = np.zeros(G+1, dtype=np.int64)  # gid=0 -> gene=0
    # fill non-NTC
    gm_nonntc = guide_map_df.loc[guide_map_df["is_ntc"] == 0, ["guide_name", "gene_name"]]
    for gname, gene in gm_nonntc.itertuples(index=False):
        gid = guide_name_to_gid[gname]
        gid_to_gene[gid] = gene_to_id[gene]

    return guide_name_to_gid, gid_to_gene, gene_names, L, G


def guides_to_padded_from_csr(Gmat_csr: sp.csr_matrix, colnames: list, guide_name_to_gid: dict, Kmax: int):
    """
    Convert CSR guide matrix to:
      guide_ids: [N,Kmax] int64 in {0..G}
      mask:      [N,Kmax] float32
    If entries have counts, pick top-Kmax by value; else arbitrary order.
    """
    N = Gmat_csr.shape[0]
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)

    indptr = Gmat_csr.indptr
    indices = Gmat_csr.indices
    data = Gmat_csr.data

    for i in range(N):
        start, end = indptr[i], indptr[i+1]
        cols = indices[start:end]
        vals = data[start:end]
        if cols.size == 0:
            continue

        # map column -> guide_id (0 for NTC if that column is NTC)
        if colnames is None:
            raise ValueError("Guide matrix has no column names; provide adata.obsm['guide'] as DataFrame or add adata.uns['guide_names'].")
        gids = np.array([guide_name_to_gid.get(colnames[c], None) for c in cols], dtype=object)
        keep = np.array([g is not None for g in gids], dtype=bool)
        cols = cols[keep]
        vals = vals[keep]
        gids = np.array([int(g) for g in gids[keep]], dtype=np.int64)

        if gids.size == 0:
            continue

        # top-K by counts if available
        if gids.size > Kmax:
            topk = np.argpartition(-vals, Kmax-1)[:Kmax]
            gids = gids[topk]

        m = gids.size
        guide_ids[i, :m] = gids
        mask[i, :m] = 1.0

    return guide_ids, mask


# ---------- Pyro model (MES/NEU logits; EC reference) ----------

def fate_model(p, day, rep, k_centered, guide_ids, mask, gene_of_guide,
               L, G, D, R, Kmax,
               s_tau=2.0, s_time=0.8, s_guide=0.8,
               s_alpha=1.0, s_rep=0.5, s_gamma=0.5,
               batch_size=8192):

    device = p.device
    N = p.shape[0]
    Fstar = 2  # MES, NEU

    tau = pyro.sample("tau", dist.HalfNormal(s_tau).expand([Fstar]).to_event(1))
    sigma_time = pyro.sample("sigma_time", dist.HalfNormal(s_time).expand([Fstar]).to_event(1))
    sigma_guide = pyro.sample("sigma_guide", dist.HalfNormal(s_guide).expand([Fstar]).to_event(1))

    alpha = pyro.sample("alpha", dist.Normal(0., s_alpha).expand([Fstar, D]).to_event(2))
    b = pyro.sample("b", dist.Normal(0., s_rep).expand([Fstar, R]).to_event(2))
    gamma = pyro.sample("gamma", dist.Normal(0., s_gamma).expand([Fstar]).to_event(1))

    # gene effects (non-centered RW)
    z0 = pyro.sample("z0", dist.Normal(0., 1.).expand([L, Fstar]).to_event(2))
    theta0 = tau.unsqueeze(0) * z0  # [L,2]
    eps = pyro.sample("eps", dist.Normal(0., 1.).expand([L, Fstar, D-1]).to_event(3))
    inc = sigma_time.unsqueeze(0).unsqueeze(-1) * eps  # [L,2,D-1]

    theta = torch.zeros((L, Fstar, D), device=device)
    theta[:, :, 0] = theta0
    theta[:, :, 1:] = theta0.unsqueeze(-1) + torch.cumsum(inc, dim=-1)
    theta = torch.cat([torch.zeros((1, Fstar, D), device=device), theta], dim=0)  # gene 0 row

    # guide deviations (time-invariant)
    u = pyro.sample("u", dist.Normal(0., 1.).expand([G, Fstar]).to_event(2))
    delta = sigma_guide.unsqueeze(0) * u
    delta = torch.cat([torch.zeros((1, Fstar), device=device), delta], dim=0)  # guide 0 row

    with pyro.plate("cells", N, subsample_size=min(batch_size, N)) as idx:
        p_b = p[idx]
        day_b = day[idx]
        rep_b = rep[idx]
        k_b = k_centered[idx]
        gids_b = guide_ids[idx]
        mask_b = mask[idx].to(p.dtype)

        gene_ids_b = gene_of_guide[gids_b]  # [B,K]

        theta_g = theta[gene_ids_b]  # [B,K,2,D]
        day_idx = day_b.view(-1, 1, 1, 1).expand(-1, Kmax, Fstar, 1)
        theta_sel = torch.gather(theta_g, dim=-1, index=day_idx).squeeze(-1)  # [B,K,2]
        delta_sel = delta[gids_b]  # [B,K,2]

        guide_contrib = (mask_b.unsqueeze(-1) * (theta_sel + delta_sel)).sum(dim=1)  # [B,2]

        alpha_b = alpha[:, day_b].transpose(0, 1)  # [B,2]
        b_b = b[:, rep_b].transpose(0, 1)          # [B,2]
        burden = k_b.view(-1, 1) * gamma.view(1, -1)

        eta_star = alpha_b + b_b + burden + guide_contrib  # [B,2]

        Bsz = eta_star.shape[0]
        eta = torch.zeros((Bsz, 3), device=device, dtype=p.dtype)
        eta[:, 1] = eta_star[:, 0]  # MES
        eta[:, 2] = eta_star[:, 1]  # NEU
        log_pi = torch.log_softmax(eta, dim=-1)

        ll = (p_b * log_pi).sum(dim=-1).sum()
        scale = float(N) / float(Bsz)
        pyro.factor("soft_ce", scale * ll)


def fit_svi(model_args, L, G, D, R, Kmax, batch_size, lr, clip_norm, num_steps, seed):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    def _model(p, day, rep, k, gids, mask, gene_of_guide):
        return fate_model(p, day, rep, k, gids, mask, gene_of_guide,
                          L=L, G=G, D=D, R=R, Kmax=Kmax, batch_size=batch_size)

    guide = AutoNormal(_model)
    optim = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = SVI(_model, guide, optim, loss=Trace_ELBO())

    for step in range(1, num_steps + 1):
        loss = svi.step(*model_args)
        if step % 200 == 0:
            print(f"step {step:5d} | loss {loss:.3f}")
    return guide


@torch.no_grad()
def reconstruct_theta_samples(samples, L, D):
    tau = samples["tau"]               # [S,2]
    sigma_time = samples["sigma_time"] # [S,2]
    z0 = samples["z0"]                 # [S,L,2]
    eps = samples["eps"]               # [S,L,2,D-1]

    theta0 = tau[:, None, :] * z0
    inc = sigma_time[:, None, :, None] * eps
    inc_cum = torch.cumsum(inc, dim=-1)

    S = theta0.shape[0]
    theta = torch.zeros((S, L, 2, D), device=theta0.device, dtype=theta0.dtype)
    theta[..., 0] = theta0
    theta[..., 1:] = theta0[..., None] + inc_cum
    return theta


@torch.no_grad()
def export_gene_summary_for_ash(guide, model_args, gene_names, L, D, num_draws, day_counts, weights, out_csv):
    samples = guide.sample_posterior(torch.Size([num_draws]), *model_args)
    theta = reconstruct_theta_samples(samples, L=L, D=D)   # [S,L,2,D], MES index=0
    theta_mes = theta[:, :, 0, :]                          # [S,L,D]

    if weights is None:
        w = torch.tensor(day_counts, device=theta_mes.device, dtype=theta_mes.dtype)
        w = w / w.sum()
    else:
        w = torch.tensor(weights, device=theta_mes.device, dtype=theta_mes.dtype)
        w = w / w.sum()

    delta_sum = (theta_mes * w[None, None, :]).sum(dim=-1)  # [S,L]
    betahat = delta_sum.mean(dim=0).cpu().numpy()
    sebetahat = delta_sum.std(dim=0, unbiased=True).cpu().numpy()

    df = pd.DataFrame({"gene": gene_names, "betahat": betahat, "sebetahat": sebetahat})
    for d in range(D):
        df[f"w{d}"] = float(w[d].cpu().item())
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--guide-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    adata = ad.read_h5ad(args.adata)

    # day, rep
    day_int = parse_day_to_int(adata.obs[cfg["day_key"]], D=cfg["D"])
    covar = adata.obsm[cfg["covar_key"]]
    if hasattr(covar, "columns"):  # DataFrame
        rep_raw = covar[cfg["rep_key"]].to_numpy()
    else:
        # structured array case:
        rep_raw = covar[cfg["rep_key"]]
    rep_int = parse_rep_to_int(rep_raw, R=cfg["R"])

    # fate probs
    p = get_fate_probs(adata, key=cfg["fate_prob_key"], fates=cfg["fates"])

    # guide matrix
    guide_obsm = adata.obsm[cfg["guide_key"]]
    if hasattr(guide_obsm, "columns"):
        guide_names = list(guide_obsm.columns)
        Gmat = guide_obsm.values
    else:
        # if not a DataFrame, try to find guide names in adata.uns
        guide_names = adata.uns.get("guide_names", None)
        Gmat = guide_obsm
    Gmat = sp.csr_matrix(Gmat)

    # mapping
    guide_map_df = load_guide_map(args.guide_map)
    guide_name_to_gid, gid_to_gene, gene_names, L, G = build_id_maps(guide_names, guide_map_df)

    # padded guide_ids/mask
    guide_ids, mask = guides_to_padded_from_csr(Gmat, guide_names, guide_name_to_gid, Kmax=cfg["Kmax"])

    # filter by k<=Kmax (should be true but keep safe)
    k = mask.sum(axis=1).astype(np.int64)
    keep = (k <= cfg["Kmax"])
    p = p[keep]
    day_int = day_int[keep]
    rep_int = rep_int[keep]
    guide_ids = guide_ids[keep]
    mask = mask[keep]

    # k centered within day
    k = mask.sum(axis=1).astype(np.float32)
    k_centered = k.copy()
    for d in np.unique(day_int):
        idx = (day_int == d)
        k_centered[idx] = k[idx] - k[idx].mean()

    # day counts for weights
    day_counts = [int((day_int == d).sum()) for d in range(cfg["D"])]

    # torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    day_t = torch.tensor(day_int, dtype=torch.long, device=device)
    rep_t = torch.tensor(rep_int, dtype=torch.long, device=device)
    k_t = torch.tensor(k_centered, dtype=torch.float32, device=device)
    gids_t = torch.tensor(guide_ids, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    gene_of_guide_t = torch.tensor(gid_to_gene, dtype=torch.long, device=device)

    model_args = (p_t, day_t, rep_t, k_t, gids_t, mask_t, gene_of_guide_t)

    guide = fit_svi(
        model_args=model_args,
        L=L, G=G, D=cfg["D"], R=cfg["R"], Kmax=cfg["Kmax"],
        batch_size=cfg["batch_size"], lr=cfg["lr"], clip_norm=cfg["clip_norm"],
        num_steps=cfg["num_steps"], seed=cfg["seed"]
    )

    export_gene_summary_for_ash(
        guide=guide,
        model_args=model_args,
        gene_names=gene_names,
        L=L, D=cfg["D"],
        num_draws=cfg["num_posterior_draws"],
        day_counts=day_counts,
        weights=cfg.get("weights", None),
        out_csv=args.out
    )


if __name__ == "__main__":
    main()
```

---

## 4) `scripts/run_ash.R` (separate, stable)

```r
#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(ashr)
})

args <- commandArgs(trailingOnly=TRUE)
in_csv  <- args[1]
out_csv <- args[2]

dt <- fread(in_csv)

fit <- ash(
  betahat   = dt$betahat,
  sebetahat = dt$sebetahat,
  method    = "shrinkage",
  mixcompdist = "halfuniform",
  outputlevel = c("PosteriorMean","PosteriorSD","lfsr","qvalue")
)

dt[, postmean := get_pm(fit)]
dt[, postsd   := get_psd(fit)]
dt[, lfsr     := get_lfsr(fit)]
dt[, qvalue   := get_qvalue(fit)]

fwrite(dt, out_csv)
```

---

## 5) `scripts/rank_hits.py`

```python
#!/usr/bin/env python
import argparse, yaml
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ash", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dt = pd.read_csv(args.ash)

    lfsr_t = cfg["lfsr_thresh"]
    q_t = cfg["qvalue_thresh"]

    dt["hit"] = (dt["lfsr"] < lfsr_t) & (dt["qvalue"] < q_t)
    dt = dt.sort_values(["hit", "lfsr"], ascending=[False, True])

    dt.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    print(dt.loc[dt["hit"]].head(20)[["gene","postmean","lfsr","qvalue"]])

if __name__ == "__main__":
    main()
```

---

## 6) Run

```bash
snakemake -j 1 --use-conda
# or if you manage envs yourself:
snakemake -j 1
```

---

### Notes you’ll care about in practice (no extra questions, just assumptions)

* This assumes `adata.obsm["guide"]` is either:

  * a **DataFrame** with columns = guide names, OR
  * a matrix plus `adata.uns["guide_names"]` listing column names.
* `guide_map.csv` must have `guide_name` matching those column names, `gene_name`, and `is_ntc` (1/0).
* We’re using **weak (non-sparse) priors** by default in `fate_model` (`s_tau=2.0`, `s_time=0.8`, `s_guide=0.8`). Adjust later if you see over/under-shrinkage.

If you want, I can add (in the Snakefile) separate conda envs per rule (`envs/pyro.yaml` and `envs/ashr.yaml`) so R and PyTorch don’t share the same env.

