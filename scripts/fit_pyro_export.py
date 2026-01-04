#!/usr/bin/env python
"""Fit the Pyro fate model and export gene summaries for mashr."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import anndata as ad  # noqa: E402
import torch  # noqa: E402

from scripts.pyro_io import (  # noqa: E402
    build_id_maps,
    load_adata_inputs,
    load_guide_map,
    normalize_config,
)
from src.models.pyro_model import (  # noqa: E402
    export_gene_summary_for_ash,
    export_gene_summary_for_mash,
    fit_svi,
    reconstruct_delta_samples,
    reconstruct_theta_samples,
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


def _get_guide_names(adata, guide_key: str) -> list[str]:
    guide_obsm = adata.obsm[guide_key]
    if hasattr(guide_obsm, "columns"):
        return list(guide_obsm.columns)
    guide_names = adata.uns.get("guide_names", None)
    if guide_names is None:
        raise ValueError("Missing guide names in adata.uns['guide_names']")
    return list(guide_names)



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
    guide_to_gene_t: torch.Tensor,
    n_guides_per_gene_t: torch.Tensor,
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

        model_args_b = (
            p_b,
            day_b,
            rep_b,
            k_b,
            gids_b,
            mask_b,
            gene_of_guide_t,
            guide_to_gene_t,
            n_guides_per_gene_t,
        )
        guide_b = fit_svi(
            p_b,
            day_b,
            rep_b,
            k_b,
            gids_b,
            mask_b,
            gene_of_guide_t,
            guide_to_gene_t,
            n_guides_per_gene_t,
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-gene", default=None)
    ap.add_argument("--out-guide", default=None)
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

    if args.out is None and args.out_gene is None and args.out_guide is None:
        raise SystemExit("Provide --out, --out-gene, or --out-guide.")
    out_gene = args.out_gene or args.out
    out_guide = args.out_guide

    logger.info("Preparing inputs")
    (
        cell_df,
        p,
        guide_ids,
        mask,
        gene_of_guide,
        guide_to_gene,
        n_guides_per_gene,
        gene_names,
        L,
        G,
        day_counts,
    ) = load_adata_inputs(adata, cfg, args.guide_map)

    k_centered = make_k_centered(cell_df, mask)

    device = _select_device(cfg)
    logger.info(f"Using device: {device}")

    (
        p_t,
        day_t,
        rep_t,
        gids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    ) = to_torch(
        cell_df,
        p,
        guide_ids,
        mask,
        gene_of_guide,
        guide_to_gene,
        n_guides_per_gene,
        device,
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
        guide_to_gene_t,
        n_guides_per_gene_t,
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

    model_args = (
        p_t,
        day_t,
        rep_t,
        k_t,
        gids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    )
    if out_gene is not None or out_guide is not None:
        logger.info("Sampling posterior draws for theta/delta summaries")
        theta_samples = reconstruct_theta_samples(
            guide=guide,
            model_args=model_args,
            L=L,
            D=cfg["D"],
            num_draws=cfg["num_posterior_draws"],
        )
        delta_samples = reconstruct_delta_samples(
            guide=guide,
            model_args=model_args,
            L=L,
            D=cfg["D"],
            num_draws=cfg["num_posterior_draws"],
        )

        _, non_ref_fates, _, _ = resolve_fate_names(cfg["fates"], ref_fate=ref_fate)
        contrast_idx = non_ref_fates.index(contrast_fate)

        theta_contrast = theta_samples[:, 1:, contrast_idx, :]
        delta_contrast = delta_samples[:, 1:, contrast_idx]
        theta_mean = theta_contrast.mean(axis=0)
        theta_sd = theta_contrast.std(axis=0, ddof=0)
        delta_mean = delta_contrast.mean(axis=0)
        delta_sd = delta_contrast.std(axis=0, ddof=0)

        out_dir = Path(out_gene or out_guide).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        guide_names = _get_guide_names(adata, cfg["guide_key"])
        guide_map_df = load_guide_map(args.guide_map)
        guide_name_to_gid, _, _, _, _ = build_id_maps(guide_names, guide_map_df)
        gid_to_guide_name = {
            gid: name for name, gid in guide_name_to_gid.items() if gid > 0
        }
        guide_names_ordered = [
            gid_to_guide_name.get(gid, f"guide_{gid}") for gid in range(1, G + 1)
        ]

        np.savez(
            out_dir / "theta_posterior_summary.npz",
            theta_mean=theta_mean,
            theta_sd=theta_sd,
            gene_names=np.asarray(gene_names, dtype=object),
        )
        np.savez(
            out_dir / "delta_posterior_summary.npz",
            delta_mean=delta_mean,
            delta_sd=delta_sd,
            guide_names=np.asarray(guide_names_ordered, dtype=object),
        )

        if np.any(n_guides_per_gene <= 0):
            raise RuntimeError("Found genes with zero guides; cannot center delta.")

        sum_delta = np.bincount(guide_to_gene, weights=delta_mean, minlength=L)
        delta_mean_gene = sum_delta / n_guides_per_gene
        delta_sd_gene = np.zeros(L, dtype=np.float64)
        for gene_idx in range(L):
            idx = guide_to_gene == gene_idx
            if idx.sum() <= 1:
                delta_sd_gene[gene_idx] = 0.0
            else:
                delta_sd_gene[gene_idx] = delta_mean[idx].std(ddof=0)

        qc_delta = pd.DataFrame(
            {
                "gene": gene_names,
                "n_guides": n_guides_per_gene,
                "delta_mean_gene": delta_mean_gene,
                "delta_sd_gene": delta_sd_gene,
            }
        )
        qc_delta.to_csv(out_dir / "qc_delta_mean_by_gene.csv", index=False)

        theta_by_guide = theta_contrast[:, guide_to_gene, :]
        beta_samples = theta_by_guide + delta_contrast[:, :, None]
        beta_mean = beta_samples.mean(axis=0)
        beta_sd = beta_samples.std(axis=0, ddof=0)

        beta_mean_gene_day = np.zeros((L, cfg["D"]), dtype=np.float64)
        for d in range(cfg["D"]):
            sum_beta = np.bincount(
                guide_to_gene, weights=beta_mean[:, d], minlength=L
            )
            beta_mean_gene_day[:, d] = sum_beta / n_guides_per_gene
        offset_day = beta_mean_gene_day - theta_mean
        offset_mean = offset_day.mean(axis=1)
        offset_sd = offset_day.std(axis=1, ddof=0)

        qc_offset = pd.DataFrame(
            {
                "gene": gene_names,
                "n_guides": n_guides_per_gene,
                "offset_mean": offset_mean,
                "offset_sd_across_days": offset_sd,
            }
        )
        qc_offset.to_csv(out_dir / "qc_theta_beta_offset_by_gene.csv", index=False)

        is_sim = "true_gene_betahat_daywise" in adata.uns
        max_delta = float(np.max(np.abs(delta_mean_gene)))
        max_offset_mean = float(np.max(np.abs(offset_mean)))
        max_offset_sd = float(np.max(offset_sd))
        thresh = 0.02
        if max_delta > thresh or max_offset_mean > thresh or max_offset_sd > thresh:
            worst_delta = qc_delta.reindex(
                qc_delta["delta_mean_gene"].abs().sort_values(ascending=False).index
            ).head(10)
            worst_offset = qc_offset.reindex(
                qc_offset["offset_mean"].abs().sort_values(ascending=False).index
            ).head(10)
            worst_offset_sd = qc_offset.reindex(
                qc_offset["offset_sd_across_days"]
                .abs()
                .sort_values(ascending=False)
                .index
            ).head(10)
            logger.warning("Delta centering check failed (max=%.4f)", max_delta)
            logger.warning("Worst delta_mean_gene:\\n%s", worst_delta.to_string(index=False))
            logger.warning("Offset check failed (max mean=%.4f, max sd=%.4f)", max_offset_mean, max_offset_sd)
            logger.warning("Worst offset_mean:\\n%s", worst_offset.to_string(index=False))
            logger.warning(
                "Worst offset_sd_across_days:\\n%s",
                worst_offset_sd.to_string(index=False),
            )
            if is_sim:
                raise RuntimeError("Delta centering QC failed in simulation mode.")

        if out_gene is not None:
            logger.info("Exporting gene summary for mashr")
            summary_df = pd.DataFrame({"gene": list(gene_names)})
            for d in range(cfg["D"]):
                summary_df[f"betahat_d{d}"] = theta_mean[:, d]
            for d in range(cfg["D"]):
                summary_df[f"se_d{d}"] = theta_sd[:, d]
            if cfg.get("bootstrap_se", False):
                logger.warning(
                    "bootstrap_se is configured but mashr export uses posterior SD per day; "
                    "bootstrap SE is skipped for mash output."
                )
            summary_df.to_csv(out_gene, index=False)
            logger.info("Wrote gene summary: %s", out_gene)

        if out_guide is not None:
            logger.info("Exporting guide summary for mashr")
            rows = []
            for gid in range(1, G + 1):
                gene_id = int(gene_of_guide[gid])
                if gene_id == 0:
                    continue
                guide_name = gid_to_guide_name.get(gid)
                if guide_name is None:
                    continue
                gene_name = gene_names[gene_id - 1]
                g_idx = gid - 1
                row = {"guide": guide_name, "gene": gene_name}
                for d in range(cfg["D"]):
                    row[f"betahat_d{d}"] = beta_mean[g_idx, d]
                for d in range(cfg["D"]):
                    row[f"se_d{d}"] = beta_sd[g_idx, d]
                rows.append(row)

            guide_out = Path(out_guide)
            guide_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(guide_out, index=False)
            logger.info("Wrote guide summary: %s", guide_out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
