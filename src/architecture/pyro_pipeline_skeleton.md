```python
"""
Full pipeline skeleton: MOI-aware CellRank fate model (Pyro) + gene-summary EB shrinkage (R ashr)

Assumptions:
- You already computed per-cell CellRank absorption probs p_i over (EC, MES, NEU).
- You have per-cell (day, rep) and detected guide IDs.
- High MOI: represent guides per cell as padded list (guide_ids, mask) with Kmax=20.
- NTC are hard-zero by mapping to guide_id=0 and gene_of_guide[0]=0.
- Fit Pyro model with SVI.
- Export gene MES–EC summary across days (betahat, sebetahat).
- Run R ashr (method="shrinkage", mixcompdist="halfuniform").
- Import results and call hits.

This file intentionally leaves "data loading" stubs for you to fill.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pyro

# ---------------------------
# CONFIG
# ---------------------------
@dataclass
class Config:
    Kmax: int = 20
    D: int = 4
    R: int = 2
    L: int = 300
    G: int = 2000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SVI
    batch_size: int = 8192
    lr: float = 1e-3
    clip_norm: float = 5.0
    num_steps_stage1: int = 1000      # nuisance-only (optional)
    num_steps_stage2: int = 5000      # full model

    # posterior draws for export
    num_draws: int = 1000

    # weights for across-day summary (if None, use day cell counts)
    weights: Optional[List[float]] = None

    # effect threshold for your own posterior diagnostics (not used by ash)
    eps_effect: float = 0.1

    # ash / hit calling
    lfsr_thresh: float = 0.05
    qvalue_thresh: float = 0.10

    # paths
    out_dir: str = "out_fate_pipeline"
    gene_summary_csv: str = "gene_summary_for_ash.csv"
    ash_out_csv: str = "gene_summary_ash_out.csv"
    run_ash_r: str = "run_ash.R"


# ---------------------------
# IO STUBS (you fill these)
# ---------------------------
def load_inputs() -> Tuple[
    pd.DataFrame,           # cell table with at least: cell_id, day, rep, k, etc.
    np.ndarray,             # p: [N,3] float32 (EC, MES, NEU)
    List[List[int]],        # guides_per_cell: list length N, each is list of detected guide IDs (raw IDs)
    pd.DataFrame,           # guide_map: columns [raw_guide_id, guide_id (1..G or 0 for NTC), gene_id (1..L or 0)]
    List[str],              # gene_names length L (genes 1..L)
]:
    raise NotImplementedError


# ---------------------------
# PREPROCESS
# ---------------------------
def build_padded_guides(
    guides_per_cell: List[List[int]],
    guide_map: pd.DataFrame,
    Kmax: int,
    G: int,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      guide_ids: [N,Kmax] int64 in {0..G}
      mask     : [N,Kmax] float32 (1 real entry incl NTC-as-0, 0 PAD)
      gene_of_guide: [G+1] int64 mapping guide_id -> gene_id in {0..L}, with gene_of_guide[0]=0

    Hard-zero convention:
      - All NTC are mapped to guide_id=0 (real entry => mask=1).
      - PAD uses guide_id=0 but mask=0.
    """
    # Build raw_guide_id -> guide_id and gene_id lookups
    # guide_id is 0 for NTC, 1..G otherwise
    raw_to_gid = dict(zip(guide_map["raw_guide_id"], guide_map["guide_id"]))
    gid_to_gene = np.zeros(G + 1, dtype=np.int64)
    gid_to_gene[0] = 0
    for _, row in guide_map.iterrows():
        gid = int(row["guide_id"])
        gene = int(row["gene_id"])
        if gid >= 0:
            gid_to_gene[gid] = gene

    N = len(guides_per_cell)
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)

    for i, raw_list in enumerate(guides_per_cell):
        gids = []
        for raw in raw_list:
            if raw in raw_to_gid:
                gids.append(int(raw_to_gid[raw]))
        # truncate to Kmax (or you can keep top by UMI, etc.)
        gids = gids[:Kmax]
        # fill
        if len(gids) > 0:
            guide_ids[i, :len(gids)] = np.array(gids, dtype=np.int64)
            mask[i, :len(gids)] = 1.0
        # rest stays PAD (guide_id=0, mask=0)

    return guide_ids, mask, gid_to_gene


def filter_cells_by_k(
    cell_df: pd.DataFrame,
    p: np.ndarray,
    guide_ids: np.ndarray,
    mask: np.ndarray,
    Kmax: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter cells with k > Kmax where k = number of real guide entries (mask==1).
    """
    k = mask.sum(axis=1).astype(np.int64)
    keep = (k <= Kmax)
    cell_df2 = cell_df.loc[keep].reset_index(drop=True)
    return cell_df2, p[keep], guide_ids[keep], mask[keep]


def make_k_centered(cell_df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """
    k centered within day: k_i - mean_k_day
    """
    k = mask.sum(axis=1).astype(np.float32)
    day = cell_df["day"].to_numpy()
    k_centered = k.copy()
    for d in np.unique(day):
        idx = (day == d)
        k_centered[idx] = k[idx] - k[idx].mean()
    return k_centered.astype(np.float32)


def to_torch(
    cell_df: pd.DataFrame,
    p: np.ndarray,
    guide_ids: np.ndarray,
    mask: np.ndarray,
    gene_of_guide: np.ndarray,
    device: str,
):
    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    day_t = torch.tensor(cell_df["day"].to_numpy(), dtype=torch.long, device=device)
    rep_t = torch.tensor(cell_df["rep"].to_numpy(), dtype=torch.long, device=device)
    guide_ids_t = torch.tensor(guide_ids, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    gene_of_guide_t = torch.tensor(gene_of_guide, dtype=torch.long, device=device)
    return p_t, day_t, rep_t, guide_ids_t, mask_t, gene_of_guide_t


# ---------------------------
# MODEL IMPORTS
# (Paste in the "fate_model", "fit_svi", "reconstruct_theta_samples",
#  and "export_gene_summary_for_ash" helpers from our previous messages)
# ---------------------------
# from your_model_file import fate_model, fit_svi, reconstruct_theta_samples, export_gene_summary_for_ash


# ---------------------------
# R (ash) RUNNER
# ---------------------------
def run_ash_rscript(in_csv: str, out_csv: str, r_script_path: str):
    cmd = ["Rscript", r_script_path, in_csv, out_csv]
    subprocess.check_call(cmd)


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) Load raw inputs
    cell_df, p, guides_per_cell, guide_map, gene_names = load_inputs()

    # 2) Build padded guides + gene_of_guide
    guide_ids, mask, gene_of_guide = build_padded_guides(
        guides_per_cell=guides_per_cell,
        guide_map=guide_map,
        Kmax=cfg.Kmax,
        G=cfg.G,
        L=cfg.L,
    )

    # 3) Filter by k<=Kmax (should be mostly true if you already truncated; still safe)
    cell_df, p, guide_ids, mask = filter_cells_by_k(cell_df, p, guide_ids, mask, cfg.Kmax)

    # 4) Build k_centered within day
    k_centered = make_k_centered(cell_df, mask)

    # 5) Day counts for weighting across-day summary
    day_counts = [int((cell_df["day"].to_numpy() == d).sum()) for d in range(cfg.D)]

    # 6) Move to torch
    p_t, day_t, rep_t, guide_ids_t, mask_t, gene_of_guide_t = to_torch(
        cell_df, p, guide_ids, mask, gene_of_guide, cfg.device
    )
    k_t = torch.tensor(k_centered, dtype=torch.float32, device=cfg.device)

    # 7) Fit Pyro model (single-stage full model; add stage1 nuisance-only if you implement it)
    model_args = (p_t, day_t, rep_t, k_t, guide_ids_t, mask_t, gene_of_guide_t)

    # guide = fit_svi(*model_args, L=cfg.L, G=cfg.G, D=cfg.D, R=cfg.R, Kmax=cfg.Kmax,
    #                 batch_size=cfg.batch_size, lr=cfg.lr, clip_norm=cfg.clip_norm,
    #                 num_steps=cfg.num_steps_stage2, seed=0)

    # (placeholder)
    guide = None

    # 8) Export gene summaries for ash
    gene_summary_path = os.path.join(cfg.out_dir, cfg.gene_summary_csv)
    # export_gene_summary_for_ash(
    #     guide=guide,
    #     model_args=model_args,
    #     gene_names=gene_names,
    #     L=cfg.L, D=cfg.D,
    #     num_draws=cfg.num_draws,
    #     day_cell_counts=day_counts,
    #     weights=cfg.weights,
    #     out_csv=gene_summary_path,
    # )

    # 9) Run R ashr
    ash_out_path = os.path.join(cfg.out_dir, cfg.ash_out_csv)
    run_ash_rscript(gene_summary_path, ash_out_path, cfg.run_ash_r)

    # 10) Import ash results + call hits
    ash = pd.read_csv(ash_out_path)
    ash["hit"] = (ash["lfsr"] < cfg.lfsr_thresh) & (ash["qvalue"] < cfg.qvalue_thresh)
    ash = ash.sort_values(["hit", "lfsr"], ascending=[False, True])

    # 11) Save hit table
    ash.to_csv(os.path.join(cfg.out_dir, "hits_ranked.csv"), index=False)
    print("Saved:", os.path.join(cfg.out_dir, "hits_ranked.csv"))
    print("Top hits:\n", ash.loc[ash["hit"]].head(20)[["gene", "postmean", "lfsr", "qvalue"]])


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
```

```r
#!/usr/bin/env Rscript
# run_ash.R
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
  method    = "shrinkage",        # pointmass=FALSE, prior="uniform"
  mixcompdist = "halfuniform",
  outputlevel = c("PosteriorMean","PosteriorSD","lfsr","qvalue")
)

dt[, postmean := get_pm(fit)]
dt[, postsd   := get_psd(fit)]
dt[, lfsr     := get_lfsr(fit)]
dt[, qvalue   := get_qvalue(fit)]

fwrite(dt, out_csv)
```

What you fill in:

* `load_inputs()` (how you load cellrank probs, day/rep, guide calls, and guide→gene mapping)
* Paste in the Pyro functions from the “rewritten skeleton” message (`fate_model`, `fit_svi`, `reconstruct_theta_samples`, `export_gene_summary_for_ash`)


