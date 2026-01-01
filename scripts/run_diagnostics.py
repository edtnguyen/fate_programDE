#!/usr/bin/env python
"""Run minimal diagnostics for the Pyro fate model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import anndata as ad  # noqa: E402
import torch  # noqa: E402
from pyro.infer import Predictive  # noqa: E402

from scripts.pyro_io import load_adata_inputs  # noqa: E402
from src.models.pyro_model import (  # noqa: E402
    add_zero_gene_row,
    add_zero_guide_row,
    compute_linear_predictor,
    construct_delta_core,
    construct_theta_core,
    export_gene_summary_for_ash,
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


def _split_indices(
    n: int, holdout_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * holdout_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=np.float64)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    q_sorted = pvals[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qvals = np.empty_like(q_sorted)
    qvals[order] = q_sorted
    return qvals


def _sanity_ok(beta: float, expected: str, min_abs: float) -> bool:
    if expected == "positive":
        return beta > min_abs
    if expected == "negative":
        return beta < -min_abs
    if expected in {"either", "any"}:
        return abs(beta) >= min_abs
    raise ValueError(f"Unsupported expected_sign '{expected}'")


def _estimate_mean_loglik(
    guide,
    model_args_train: tuple,
    p_test: torch.Tensor,
    day_test: torch.Tensor,
    rep_test: torch.Tensor,
    k_test: torch.Tensor,
    gids_test: torch.Tensor,
    mask_test: torch.Tensor,
    gene_of_guide_t: torch.Tensor,
    *,
    fate_names: Sequence[str],
    ref_fate: str,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    num_draws: int,
) -> float:
    fate_names, _, _, non_ref_indices = resolve_fate_names(
        fate_names, ref_fate=ref_fate
    )
    return_sites = [
        "alpha",
        "b",
        "gamma",
        "tau",
        "z0",
        "sigma_time",
        "sigma_guide",
        "u",
    ]
    if D > 1:
        return_sites.append("eps")
    predictive = Predictive(guide, num_samples=num_draws, return_sites=return_sites)
    samples = predictive(
        *model_args_train,
        L=L,
        G=G,
        D=D,
        R=R,
        Kmax=Kmax,
    )

    theta_core = construct_theta_core(
        tau=samples["tau"],
        z0=samples["z0"],
        sigma_time=samples["sigma_time"],
        eps=samples.get("eps"),
        D=D,
    )
    theta = add_zero_gene_row(theta_core)
    delta_core = construct_delta_core(
        sigma_guide=samples["sigma_guide"], u=samples["u"]
    )
    delta = add_zero_guide_row(delta_core)

    logliks = []
    for s in range(theta.shape[0]):
        eta_nonref = compute_linear_predictor(
            alpha_t=samples["alpha"][s],
            b_t=samples["b"][s],
            gamma_t=samples["gamma"][s],
            k_t=k_test,
            guide_ids_t=gids_test,
            mask_t=mask_test,
            gene_of_guide_t=gene_of_guide_t,
            theta_t=theta[s],
            delta_t=delta[s],
            day_t=day_test,
            rep_t=rep_test,
        )
        eta = torch.zeros((p_test.shape[0], len(fate_names)), device=p_test.device)
        eta[:, non_ref_indices] = eta_nonref
        log_pi = torch.log_softmax(eta, dim=-1)
        logp = (p_test * log_pi).sum(-1)
        logliks.append(logp.mean().item())

    return float(np.mean(logliks))


def _permute_guides(
    guide_ids: np.ndarray,
    mask: np.ndarray,
    day: np.ndarray,
    rep: np.ndarray,
    seed: int,
) -> np.ndarray:
    gids_perm = guide_ids.copy()
    rng = np.random.default_rng(seed)
    k = mask.sum(axis=1).astype(np.int64)
    keys = list(zip(day, rep, k))
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        gids_perm[idx] = guide_ids[perm]
    return gids_perm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--guide-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    cfg = yaml.safe_load(open(args.config))
    ref_fate = cfg.get("ref_fate", "EC")
    contrast_fate = cfg.get("contrast_fate", "MES")
    _, non_ref_fates, _, _ = resolve_fate_names(cfg["fates"], ref_fate=ref_fate)
    if contrast_fate not in non_ref_fates:
        raise ValueError(
            f"contrast_fate '{contrast_fate}' not in non-reference fates {non_ref_fates}"
        )
    adata = ad.read_h5ad(args.adata)

    (
        cell_df,
        p,
        guide_ids,
        mask,
        gene_of_guide,
        gene_names,
        L,
        G,
        _,
    ) = load_adata_inputs(adata, cfg, args.guide_map)

    k_centered = make_k_centered(cell_df, mask)

    n_cells = len(cell_df)
    train_idx, test_idx = _split_indices(
        n_cells,
        cfg.get("diagnostics_holdout_frac", 0.1),
        cfg.get("diagnostics_seed", 0),
    )

    device = _select_device(cfg)
    logger.info(f"Diagnostics using device: {device}")

    cell_df_train = cell_df.iloc[train_idx].reset_index(drop=True)
    cell_df_test = cell_df.iloc[test_idx].reset_index(drop=True)

    p_train = p[train_idx]
    p_test = p[test_idx]
    gids_train = guide_ids[train_idx]
    gids_test = guide_ids[test_idx]
    mask_train = mask[train_idx]
    mask_test = mask[test_idx]
    k_train = k_centered[train_idx]
    k_test = k_centered[test_idx]

    p_t_train, day_t_train, rep_t_train, gids_t_train, mask_t_train, gene_of_guide_t = (
        to_torch(cell_df_train, p_train, gids_train, mask_train, gene_of_guide, device)
    )
    p_t_test, day_t_test, rep_t_test, gids_t_test, mask_t_test, _ = to_torch(
        cell_df_test, p_test, gids_test, mask_test, gene_of_guide, device
    )
    k_t_train = torch.tensor(k_train, dtype=torch.float32, device=device)
    k_t_test = torch.tensor(k_test, dtype=torch.float32, device=device)

    day_counts_train = [
        int((cell_df_train["day"].to_numpy() == d).sum()) for d in range(cfg["D"])
    ]

    model_args_train = (
        p_t_train,
        day_t_train,
        rep_t_train,
        k_t_train,
        gids_t_train,
        mask_t_train,
        gene_of_guide_t,
    )

    diag_steps = cfg.get("diagnostics_num_steps", 1000)
    diag_draws = cfg.get("diagnostics_num_draws", 25)

    logger.info("Fitting full model for diagnostics")
    guide_full = fit_svi(
        p_t_train,
        day_t_train,
        rep_t_train,
        k_t_train,
        gids_t_train,
        mask_t_train,
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
        num_steps=diag_steps,
        seed=cfg.get("diagnostics_seed", 0),
    )

    full_loglik = _estimate_mean_loglik(
        guide_full,
        model_args_train,
        p_t_test,
        day_t_test,
        rep_t_test,
        k_t_test,
        gids_t_test,
        mask_t_test,
        gene_of_guide_t,
        fate_names=cfg["fates"],
        ref_fate=ref_fate,
        L=L,
        G=G,
        D=cfg["D"],
        R=cfg["R"],
        Kmax=cfg["Kmax"],
        num_draws=diag_draws,
    )

    logger.info("Fitting nuisance-only model for diagnostics")
    gids_t_train_zero = torch.zeros_like(gids_t_train)
    mask_t_train_zero = torch.zeros_like(mask_t_train)
    gids_t_test_zero = torch.zeros_like(gids_t_test)
    mask_t_test_zero = torch.zeros_like(mask_t_test)

    guide_nuisance = fit_svi(
        p_t_train,
        day_t_train,
        rep_t_train,
        k_t_train,
        gids_t_train_zero,
        mask_t_train_zero,
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
        num_steps=diag_steps,
        seed=cfg.get("diagnostics_seed", 0),
    )

    nuisance_loglik = _estimate_mean_loglik(
        guide_nuisance,
        model_args_train,
        p_t_test,
        day_t_test,
        rep_t_test,
        k_t_test,
        gids_t_test_zero,
        mask_t_test_zero,
        gene_of_guide_t,
        fate_names=cfg["fates"],
        ref_fate=ref_fate,
        L=L,
        G=G,
        D=cfg["D"],
        R=cfg["R"],
        Kmax=cfg["Kmax"],
        num_draws=diag_draws,
    )

    diagnostics = {
        "holdout_frac": float(cfg.get("diagnostics_holdout_frac", 0.1)),
        "num_test_cells": int(len(test_idx)),
        "full_model_ce": float(-full_loglik),
        "nuisance_model_ce": float(-nuisance_loglik),
        "ce_improvement": float(full_loglik - nuisance_loglik),
    }

    if cfg.get("diagnostics_run_permutation", False):
        logger.info("Running guide permutation diagnostic")
        gids_perm = _permute_guides(
            guide_ids,
            mask,
            cell_df["day"].to_numpy(),
            cell_df["rep"].to_numpy(),
            cfg.get("diagnostics_seed", 0),
        )

        gids_perm_train = gids_perm[train_idx]
        gids_perm_test = gids_perm[test_idx]

        gids_t_perm_train = torch.tensor(
            gids_perm_train, dtype=torch.long, device=device
        )
        gids_t_perm_test = torch.tensor(gids_perm_test, dtype=torch.long, device=device)

        guide_perm = fit_svi(
            p_t_train,
            day_t_train,
            rep_t_train,
            k_t_train,
            gids_t_perm_train,
            mask_t_train,
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
            num_steps=cfg.get("diagnostics_perm_steps", 500),
            seed=cfg.get("diagnostics_seed", 0),
        )

        model_args_perm = (
            p_t_train,
            day_t_train,
            rep_t_train,
            k_t_train,
            gids_t_perm_train,
            mask_t_train,
            gene_of_guide_t,
        )

        perm_summary = Path(args.out).with_name("diagnostics_perm_summary.csv")
        export_gene_summary_for_ash(
            guide=guide_perm,
            model_args=model_args_perm,
            gene_names=gene_names,
            fate_names=cfg["fates"],
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            L=L,
            D=cfg["D"],
            num_draws=cfg.get("diagnostics_perm_draws", 25),
            day_cell_counts=day_counts_train,
            weights=cfg.get("weights", None),
            out_csv=str(perm_summary),
        )

        perm_ash = Path(args.out).with_name("diagnostics_perm_ash.csv")
        perm_df = pd.read_csv(perm_summary)

        betahat = perm_df["betahat"].to_numpy(dtype=np.float64)
        sebetahat = perm_df["sebetahat"].to_numpy(dtype=np.float64)
        valid = (sebetahat > 0) & np.isfinite(sebetahat) & np.isfinite(betahat)

        z = np.zeros_like(betahat)
        z[valid] = betahat[valid] / sebetahat[valid]
        # Use two-sided normal p-values as an lfsr proxy for permutation diagnostics.
        pvals = np.ones_like(z)
        pvals[valid] = 2.0 * norm.sf(np.abs(z[valid]))

        perm_df["lfsr"] = pvals
        perm_df["qvalue"] = _bh_qvalues(pvals)
        perm_df.to_csv(perm_ash, index=False)

        hits = (perm_df["lfsr"] < cfg["lfsr_thresh"]) & (
            perm_df["qvalue"] < cfg["qvalue_thresh"]
        )
        diagnostics["perm_hit_count"] = int(hits.sum())
        diagnostics["perm_hit_frac"] = float(hits.mean())

    sanity_genes = cfg.get("sanity_check_genes", [])
    if sanity_genes:
        logger.info("Running sanity checks for %d genes", len(sanity_genes))
        sanity_df = export_gene_summary_for_ash(
            guide=guide_full,
            model_args=model_args_train,
            gene_names=gene_names,
            fate_names=cfg["fates"],
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            L=L,
            D=cfg["D"],
            num_draws=cfg.get("sanity_num_draws", 200),
            day_cell_counts=day_counts_train,
            weights=cfg.get("weights", None),
            out_csv=None,
        )
        betahat_map = dict(zip(sanity_df["gene"], sanity_df["betahat"]))
        min_abs = float(cfg.get("sanity_min_abs_effect", 0.0))
        results = []
        for entry in sanity_genes:
            if isinstance(entry, str):
                gene = entry
                expected = "either"
            elif isinstance(entry, dict):
                gene = entry.get("gene")
                expected = entry.get("expected_sign", "either")
            else:
                raise ValueError(
                    "sanity_check_genes entries must be strings or dicts with 'gene'."
                )
            if not gene:
                raise ValueError("sanity_check_genes entries must specify a gene name")
            beta = betahat_map.get(gene)
            if beta is None:
                results.append(
                    {
                        "gene": gene,
                        "expected_sign": expected,
                        "status": "missing",
                    }
                )
                continue
            results.append(
                {
                    "gene": gene,
                    "expected_sign": expected,
                    "betahat": float(beta),
                    "ok": bool(_sanity_ok(float(beta), expected, min_abs)),
                }
            )
        diagnostics["sanity_checks"] = results

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, sort_keys=True)

    logger.info("Wrote diagnostics to %s", args.out)


if __name__ == "__main__":
    main()
