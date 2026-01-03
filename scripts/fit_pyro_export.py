#!/usr/bin/env python
"""Fit the Pyro fate model and export gene summaries for mashr."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import anndata as ad  # noqa: E402
import torch  # noqa: E402

from scripts.pyro_io import load_adata_inputs, normalize_config  # noqa: E402
from src.models.pyro_model import (  # noqa: E402
    export_gene_summary_for_ash,
    export_gene_summary_for_mash,
    fit_svi,
    resolve_fate_names,
)
from src.models.pyro_pipeline import make_k_centered, to_torch  # noqa: E402


def _select_device(cfg: dict) -> str:
    requested = cfg.get("device", None)
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def _stratified_bootstrap_indices(
    day_values: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    idx = []
    for d in np.unique(day_values):
        day_idx = np.flatnonzero(day_values == d)
        if day_idx.size == 0:
            continue
        n_draw = max(1, int(np.ceil(day_idx.size * frac)))
        idx.append(rng.choice(day_idx, size=n_draw, replace=True))
    return np.concatenate(idx)


def _bootstrap_se(
    *,
    cell_day: np.ndarray,
    p_t: torch.Tensor,
    day_t: torch.Tensor,
    rep_t: torch.Tensor,
    k_t: torch.Tensor,
    gids_t: torch.Tensor,
    mask_t: torch.Tensor,
    gene_of_guide_t: torch.Tensor,
    gene_names: list[str],
    fate_names: list[str],
    ref_fate: str,
    contrast_fate: str,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    s_alpha: float,
    s_rep: float,
    s_gamma: float,
    s_tau: float,
    s_time: float,
    s_guide: float,
    likelihood_weight: float,
    num_steps: int,
    num_draws: int,
    weights: list[float] | None,
    reps: int,
    frac: float,
    seed: int,
    device: str,
    logger: logging.Logger,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    betahat_samples = []

    for rep in range(reps):
        idx = _stratified_bootstrap_indices(cell_day, frac, rng)
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)

        p_b = p_t.index_select(0, idx_t)
        day_b = day_t.index_select(0, idx_t)
        rep_b = rep_t.index_select(0, idx_t)
        k_b = k_t.index_select(0, idx_t)
        gids_b = gids_t.index_select(0, idx_t)
        mask_b = mask_t.index_select(0, idx_t)

        model_args_b = (p_b, day_b, rep_b, k_b, gids_b, mask_b, gene_of_guide_t)
        guide_b = fit_svi(
            p_b,
            day_b,
            rep_b,
            k_b,
            gids_b,
            mask_b,
            gene_of_guide_t,
            fate_names=fate_names,
            ref_fate=ref_fate,
            L=L,
            G=G,
            D=D,
            R=R,
            Kmax=Kmax,
            batch_size=batch_size,
            lr=lr,
            clip_norm=clip_norm,
            num_steps=num_steps,
            s_alpha=s_alpha,
            s_rep=s_rep,
            s_gamma=s_gamma,
            s_tau=s_tau,
            s_time=s_time,
            s_guide=s_guide,
            likelihood_weight=likelihood_weight,
            seed=seed + rep + 1,
        )

        day_counts_b = np.bincount(day_b.cpu().numpy(), minlength=D).tolist()
        summary_b = export_gene_summary_for_ash(
            guide=guide_b,
            model_args=model_args_b,
            gene_names=gene_names,
            fate_names=fate_names,
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            L=L,
            D=D,
            num_draws=num_draws,
            day_cell_counts=day_counts_b,
            weights=weights,
            out_csv=None,
        )
        betahat_samples.append(summary_b["betahat"].to_numpy())
        logger.info("Bootstrap replicate %d/%d complete", rep + 1, reps)

    betahat_stack = np.stack(betahat_samples, axis=0)
    return betahat_stack.std(axis=0, ddof=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--guide-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    cfg = normalize_config(yaml.safe_load(open(args.config)))
    ref_fate = cfg.get("ref_fate", "EC")
    contrast_fate = cfg.get("contrast_fate", "MES")
    _, non_ref_fates, _, _ = resolve_fate_names(cfg["fates"], ref_fate=ref_fate)
    if contrast_fate not in non_ref_fates:
        raise ValueError(
            f"contrast_fate '{contrast_fate}' not in non-reference fates {non_ref_fates}"
        )

    logger.info("Loading AnnData")
    adata = ad.read_h5ad(args.adata)

    logger.info("Preparing inputs")
    (
        cell_df,
        p,
        guide_ids,
        mask,
        gene_of_guide,
        gene_names,
        L,
        G,
        day_counts,
    ) = load_adata_inputs(adata, cfg, args.guide_map)

    k_centered = make_k_centered(cell_df, mask)

    device = _select_device(cfg)
    logger.info(f"Using device: {device}")

    p_t, day_t, rep_t, gids_t, mask_t, gene_of_guide_t = to_torch(
        cell_df, p, guide_ids, mask, gene_of_guide, device
    )
    k_t = torch.tensor(k_centered, dtype=torch.float32, device=device)

    logger.info("Fitting SVI")
    guide = fit_svi(
        p_t,
        day_t,
        rep_t,
        k_t,
        gids_t,
        mask_t,
        gene_of_guide_t,
        fate_names=cfg["fates"],
        ref_fate=ref_fate,
        L=L,
        G=G,
        D=cfg["D"],
        R=cfg["R"],
        Kmax=cfg["Kmax"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        clip_norm=cfg["clip_norm"],
        num_steps=cfg["num_steps"],
        s_alpha=cfg.get("s_alpha", 1.0),
        s_rep=cfg.get("s_rep", 1.0),
        s_gamma=cfg.get("s_gamma", 1.0),
        s_tau=cfg.get("s_tau", 1.0),
        s_time=cfg.get("s_time", 1.0),
        s_guide=cfg.get("s_guide", 1.0),
        likelihood_weight=cfg.get("likelihood_weight", 1.0),
        seed=cfg.get("seed", 0),
    )

    model_args = (p_t, day_t, rep_t, k_t, gids_t, mask_t, gene_of_guide_t)
    logger.info("Exporting gene summary for mashr")
    summary_df = export_gene_summary_for_mash(
        guide=guide,
        model_args=model_args,
        gene_names=gene_names,
        fate_names=cfg["fates"],
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
        L=L,
        D=cfg["D"],
        num_draws=cfg["num_posterior_draws"],
        out_csv=None,
    )

    if cfg.get("bootstrap_se", False):
        logger.warning(
            "bootstrap_se is configured but mashr export uses posterior SD per day; "
            "bootstrap SE is skipped for mash output."
        )

    summary_df.to_csv(args.out, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
