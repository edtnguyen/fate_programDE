#!/usr/bin/env python
"""Fit the Pyro fate model and export gene summaries for ash."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import anndata as ad  # noqa: E402
import torch  # noqa: E402

from scripts.pyro_io import load_adata_inputs  # noqa: E402
from src.models.pyro_model import export_gene_summary_for_ash, fit_svi  # noqa: E402
from src.models.pyro_pipeline import make_k_centered, to_torch  # noqa: E402


def _select_device(cfg: dict) -> str:
    requested = cfg.get("device", None)
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--guide-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    cfg = yaml.safe_load(open(args.config))

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
        L=L,
        G=G,
        D=cfg["D"],
        R=cfg["R"],
        Kmax=cfg["Kmax"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        clip_norm=cfg["clip_norm"],
        num_steps=cfg["num_steps"],
        seed=cfg.get("seed", 0),
    )

    model_args = (p_t, day_t, rep_t, k_t, gids_t, mask_t, gene_of_guide_t)
    logger.info("Exporting gene summary for ash")
    export_gene_summary_for_ash(
        guide=guide,
        model_args=model_args,
        gene_names=gene_names,
        L=L,
        D=cfg["D"],
        num_draws=cfg["num_posterior_draws"],
        day_cell_counts=day_counts,
        weights=cfg.get("weights", None),
        out_csv=args.out,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
