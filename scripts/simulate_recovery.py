#!/usr/bin/env python
"""Simulate known parameters and test recovery with the Pyro fate model."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch  # noqa: E402
import pyro  # noqa: E402

from src.models.pyro_model import (  # noqa: E402
    add_zero_gene_row,
    add_zero_guide_row,
    compute_linear_predictor,
    construct_delta_core,
    construct_theta_core,
    export_gene_summary_for_mash,
    fit_svi,
    resolve_fate_names,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite.")


def _parse_time_scale(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return []
    return [float(v) for v in items]


def _center_k(k_raw: np.ndarray, day: np.ndarray) -> np.ndarray:
    k_centered = k_raw.astype(np.float32)
    for d in np.unique(day):
        idx = day == d
        if idx.any():
            k_centered[idx] = k_raw[idx] - k_raw[idx].mean()
    return k_centered.astype(np.float32)


def _guide_name_lists(G: int, ntc_guides: int) -> tuple[list[str], list[str]]:
    non_ntc = [f"g{i + 1}" for i in range(G)]
    ntc = [f"ntc{i + 1}" for i in range(ntc_guides)]
    return non_ntc, ntc


def _make_gene_of_guide(G: int, L: int) -> np.ndarray:
    gene_of_guide = np.zeros(G + 1, dtype=np.int64)
    gene_of_guide[1:] = (np.arange(1, G + 1) - 1) % L + 1
    return gene_of_guide


def _simulate_parameters(
    rng: np.random.Generator,
    L: int,
    G: int,
    D: int,
    R: int,
    fstar: int,
    active_gene_frac: float,
    active_gene_min_effect: float,
    active_gene_max_effect: float,
) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    params["alpha"] = rng.normal(0.0, 0.2, size=(fstar, D)).astype(np.float32)
    params["b"] = rng.normal(0.0, 0.2, size=(fstar, R)).astype(np.float32)
    params["gamma"] = rng.normal(0.0, 0.1, size=(fstar,)).astype(np.float32)

    params["tau"] = np.full((fstar,), 0.3, dtype=np.float32)
    if D > 1:
        params["sigma_time"] = np.full((fstar, D - 1), 0.15, dtype=np.float32)
    else:
        params["sigma_time"] = np.zeros((fstar, 0), dtype=np.float32)
    params["sigma_guide"] = np.full((fstar,), 0.2, dtype=np.float32)

    # Simulate theta directly, then back out z0 and eps.
    theta_true = np.zeros((L, fstar, D), dtype=np.float32)
    n_active = int(L * active_gene_frac)
    for d in range(D):
        active_genes = rng.choice(L, n_active, replace=False)
        effects = rng.uniform(
            active_gene_min_effect, active_gene_max_effect, n_active
        )
        signs = rng.choice([-1, 1], n_active)
        for i, gene_idx in enumerate(active_genes):
            for f in range(fstar):
                theta_true[gene_idx, f, d] = effects[i] * signs[i]

    params["theta_true"] = theta_true
    params["z0"] = theta_true[:, :, 0] / params["tau"][None, :]
    if D > 1:
        eps_unscaled = np.diff(theta_true, axis=2)
        params["eps"] = eps_unscaled / params["sigma_time"][None, :, :]
    else:
        params["eps"] = None

    params["u"] = rng.normal(0.0, 1.0, size=(G, fstar)).astype(np.float32)
    return params


def _simulate_inputs(
    rng: np.random.Generator,
    *,
    N: int,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    ntc_guides: int,
    ntc_frac: float,
    fate_names: tuple[str, ...],
    ref_fate: str,
    concentration: float,
    time_scale: list[float] | None = None,
    active_gene_frac: float,
    active_gene_min_effect: float,
    active_gene_max_effect: float,
) -> tuple[
    tuple[torch.Tensor, ...],
    dict[str, np.ndarray],
    list[str],
    list[int],
    dict[str, np.ndarray],
]:
    fate_names, non_ref_fates, _, non_ref_indices = resolve_fate_names(
        fate_names, ref_fate=ref_fate
    )
    fstar = len(non_ref_fates)

    day = rng.integers(0, D, size=N, dtype=np.int64)
    rep = rng.integers(0, R, size=N, dtype=np.int64)

    max_k = max(1, Kmax)
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)
    k_raw = np.zeros(N, dtype=np.int64)
    for i in range(N):
        k_total = rng.integers(1, max_k + 1)
        if ntc_guides > 0:
            k_ntc = rng.binomial(k_total, ntc_frac)
        else:
            k_ntc = 0
        k_non = k_total - k_ntc
        if k_non > G:
            k_non = G
            k_ntc = k_total - k_non
        gids = []
        if k_non > 0:
            gids.extend(rng.choice(G, size=k_non, replace=False) + 1)
        if k_ntc > 0:
            gids.extend([0] * k_ntc)
        rng.shuffle(gids)
        guide_ids[i, : k_total] = gids
        mask[i, : k_total] = 1.0
        k_raw[i] = k_total

    gene_of_guide = _make_gene_of_guide(G, L)
    guide_to_gene = gene_of_guide[1:] - 1
    n_guides_per_gene = np.bincount(guide_to_gene, minlength=L).astype(np.int64)

    params = _simulate_parameters(
        rng,
        L=L,
        G=G,
        D=D,
        R=R,
        fstar=fstar,
        active_gene_frac=active_gene_frac,
        active_gene_min_effect=active_gene_min_effect,
        active_gene_max_effect=active_gene_max_effect,
    )
    if time_scale is not None and D > 1:
        scale = np.asarray(time_scale, dtype=np.float32)
        params["sigma_time"] = params["sigma_time"] * scale[None, :]

    theta_core = torch.tensor(params["theta_true"])
    theta = add_zero_gene_row(theta_core)

    delta_core = construct_delta_core(
        sigma_guide=torch.tensor(params["sigma_guide"]),
        u=torch.tensor(params["u"]),
        guide_to_gene=torch.tensor(guide_to_gene),
        n_guides_per_gene=torch.tensor(n_guides_per_gene),
    )
    delta = add_zero_guide_row(delta_core)

    k_centered = _center_k(k_raw.astype(np.float32), day)

    eta_nonref = compute_linear_predictor(
        alpha_t=torch.tensor(params["alpha"]),
        b_t=torch.tensor(params["b"]),
        gamma_t=torch.tensor(params["gamma"]),
        k_t=torch.tensor(k_centered),
        guide_ids_t=torch.tensor(guide_ids),
        mask_t=torch.tensor(mask),
        gene_of_guide_t=torch.tensor(gene_of_guide),
        theta_t=theta,
        delta_t=delta,
        day_t=torch.tensor(day),
        rep_t=torch.tensor(rep),
    )

    eta = torch.zeros((N, len(fate_names)), dtype=torch.float32)
    eta[:, non_ref_indices] = eta_nonref
    pi = torch.softmax(eta, dim=-1).cpu().numpy()

    if concentration <= 0:
        p = pi
    else:
        alpha = np.clip(pi, 1e-6, 1.0) * concentration
        p = np.vstack([rng.dirichlet(a) for a in alpha]).astype(np.float32)

    p_t = torch.tensor(p, dtype=torch.float32)
    day_t = torch.tensor(day, dtype=torch.long)
    rep_t = torch.tensor(rep, dtype=torch.long)
    k_t = torch.tensor(k_centered, dtype=torch.float32)
    gids_t = torch.tensor(guide_ids, dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    gene_of_guide_t = torch.tensor(gene_of_guide, dtype=torch.long)
    guide_to_gene_t = torch.tensor(guide_to_gene, dtype=torch.long)
    n_guides_per_gene_t = torch.tensor(n_guides_per_gene, dtype=torch.long)

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
    gene_names = [f"Gene{idx + 1}" for idx in range(L)]
    day_counts = np.bincount(day, minlength=D).tolist()
    sim_payload = {
        "p": p,
        "day": day,
        "rep": rep,
        "guide_ids": guide_ids,
        "mask": mask,
        "k_raw": k_raw,
    }
    return model_args, params, gene_names, day_counts, sim_payload


def _true_gene_daywise(
    params: dict[str, np.ndarray],
    *,
    L: int,
    D: int,
    fate_names: tuple[str, ...],
    ref_fate: str,
    contrast_fate: str,
) -> np.ndarray:
    _, non_ref_fates, _, _ = resolve_fate_names(fate_names, ref_fate=ref_fate)
    contrast_idx = non_ref_fates.index(contrast_fate)

    theta_core = torch.tensor(params["theta_true"])
    theta = add_zero_gene_row(theta_core).detach().cpu().numpy()
    return theta[1:, contrast_idx, :]


def _true_gene_summary(
    params: dict[str, np.ndarray],
    *,
    L: int,
    D: int,
    fate_names: tuple[str, ...],
    ref_fate: str,
    contrast_fate: str,
    day_counts: list[int],
) -> np.ndarray:
    theta_gene = _true_gene_daywise(
        params,
        L=L,
        D=D,
        fate_names=fate_names,
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
    )
    weights = np.asarray(day_counts, dtype=np.float64)
    weights = weights / weights.sum()
    return (theta_gene * weights[None, :]).sum(axis=1)


def _compute_metrics(est: np.ndarray, true: np.ndarray) -> dict[str, float]:
    est_flat = np.asarray(est, dtype=np.float64).ravel()
    true_flat = np.asarray(true, dtype=np.float64).ravel()
    rmse = float(np.sqrt(np.mean((est_flat - true_flat) ** 2)))
    corr = (
        float(np.corrcoef(est_flat, true_flat)[0, 1])
        if est_flat.size > 1
        else float("nan")
    )
    sign_acc = float(np.mean(np.sign(est_flat) == np.sign(true_flat)))
    return {"rmse": rmse, "corr": corr, "sign_acc": sign_acc}


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"{label} recovery metrics")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Pearson r: {metrics['corr']:.4f}")
    print(f"  Sign agreement: {metrics['sign_acc']:.3f}")


def _write_guide_map(
    path: Path,
    *,
    guide_names: list[str],
    gene_names: list[str],
    G: int,
) -> None:
    non_ntc = guide_names[:G]
    ntc = guide_names[G:]
    gene_map = [gene_names[i % len(gene_names)] for i in range(G)]

    guide_map = pd.DataFrame(
        {
            "guide_name": non_ntc + ntc,
            "gene_name": gene_map + ["NTC"] * len(ntc),
            "is_ntc": [0] * len(non_ntc) + [1] * len(ntc),
        }
    )
    guide_map.to_csv(path, index=False)


def _write_anndata(
    path: Path,
    *,
    fate_key: str,
    guide_key: str,
    covar_key: str,
    rep_key: str,
    day_key: str,
    fate_names: tuple[str, ...],
    gene_names: list[str],
    guide_names: list[str],
    G: int,
    sim_payload: dict[str, np.ndarray],
    true_betahat_daywise: np.ndarray,
    seed: int,
) -> None:
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - runtime guard
        raise SystemExit(f"anndata is required to write AnnData: {exc}")

    rng = np.random.default_rng(seed + 1)
    p = sim_payload["p"]
    day = sim_payload["day"]
    rep = sim_payload["rep"]
    guide_ids = sim_payload["guide_ids"]
    mask = sim_payload["mask"]

    n_cells = p.shape[0]
    cell_ids = [f"cell{i}" for i in range(n_cells)]
    day_values = [f"d{d}" for d in day]
    rep_values = [f"r{r}" for r in rep]

    obs = pd.DataFrame({day_key: day_values}, index=cell_ids)
    covar = pd.DataFrame({rep_key: rep_values}, index=cell_ids)
    fate_df = pd.DataFrame(p, index=cell_ids, columns=list(fate_names))

    guide_matrix = np.zeros((n_cells, len(guide_names)), dtype=np.int64)
    ntc_count = len(guide_names) - G
    for i in range(n_cells):
        gids = guide_ids[i][mask[i] > 0]
        for gid in gids:
            if gid == 0:
                if ntc_count == 0:
                    continue
                ntc_idx = rng.integers(0, ntc_count)
                col = G + ntc_idx
            else:
                col = gid - 1
            guide_matrix[i, col] = 1
    guide_df = pd.DataFrame(guide_matrix, index=cell_ids, columns=guide_names)

    adata = ad.AnnData(X=np.zeros((n_cells, 1), dtype=np.float32), obs=obs)
    adata.obsm[covar_key] = covar
    adata.obsm[fate_key] = fate_df
    adata.obsm[guide_key] = guide_df
    adata.uns["true_gene_betahat_daywise"] = true_betahat_daywise.tolist()
    adata.uns["gene_names"] = list(gene_names)

    adata.write_h5ad(path)


def _write_config(
    path: Path,
    *,
    adata_path: str,
    guide_map_csv: str,
    out_dir: str,
    fate_key: str,
    guide_key: str,
    covar_key: str,
    rep_key: str,
    day_key: str,
    fate_names: tuple[str, ...],
    ref_fate: str,
    contrast_fate: str,
    Kmax: int,
    D: int,
    R: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    num_steps: int,
    s_alpha: float,
    s_rep: float,
    s_gamma: float,
    s_time: float,
    s_guide: float,
    s_tau: float,
    num_draws: int,
    seed: int,
    likelihood_weight: float,
    time_scale: list[float] | None = None,
) -> None:
    cfg = {
        "adata_path": adata_path,
        "fate_prob_key": fate_key,
        "fates": list(fate_names),
        "ref_fate": ref_fate,
        "contrast_fate": contrast_fate,
        "guide_key": guide_key,
        "covar_key": covar_key,
        "rep_key": rep_key,
        "day_key": day_key,
        "guide_map_csv": guide_map_csv,
        "out_dir": out_dir,
        "Kmax": Kmax,
        "D": D,
        "R": R,
        "batch_size": batch_size,
        "lr": lr,
        "clip_norm": clip_norm,
        "num_steps": num_steps,
        "s_alpha": s_alpha,
        "s_rep": s_rep,
        "s_gamma": s_gamma,
        "s_time": s_time,
        "s_guide": s_guide,
        "s_tau": s_tau,
        "time_scale": time_scale,
        "likelihood_weight": likelihood_weight,
        "num_posterior_draws": num_draws,
        "seed": seed,
        "weights": None,
        "lfsr_thresh": 0.05,
        "qvalue_thresh": 0.10,
    }
    with path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def _write_sim_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    fate_names: tuple[str, ...],
    ref_fate: str,
    contrast_fate: str,
    day_counts: list[int],
    sim_payload: dict[str, np.ndarray],
    time_scale: list[float] | None,
) -> None:
    k_raw = sim_payload["k_raw"]
    rep_counts = np.bincount(sim_payload["rep"], minlength=args.reps)
    metadata = {
        "seed": args.seed,
        "cells": args.cells,
        "genes": args.genes,
        "guides_non_ntc": args.guides,
        "guides_ntc": args.ntc_guides,
        "guides_total": args.guides + args.ntc_guides,
        "days": args.days,
        "reps": args.reps,
        "kmax": args.kmax,
        "ntc_frac": args.ntc_frac,
        "concentration": args.concentration,
        "s_time": args.s_time,
        "s_guide": args.s_guide,
        "s_tau": args.s_tau,
        "time_scale": time_scale,
        "fates": list(fate_names),
        "ref_fate": ref_fate,
        "contrast_fate": contrast_fate,
        "day_counts": {f"d{d}": int(c) for d, c in enumerate(day_counts)},
        "rep_counts": {f"r{r}": int(c) for r, c in enumerate(rep_counts)},
        "k_summary": {
            "min": int(k_raw.min()) if k_raw.size else 0,
            "max": int(k_raw.max()) if k_raw.size else 0,
            "mean": float(k_raw.mean()) if k_raw.size else 0.0,
        },
        "adata_path": args.adata_out,
        "guide_map_csv": args.guide_map_out,
        "out_dir": args.out_dir,
    }
    with path.open("w") as fh:
        yaml.safe_dump(metadata, fh, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--genes", type=int, default=30)
    ap.add_argument("--guides", type=int, default=90)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--ntc-guides", type=int, default=1)
    ap.add_argument("--ntc-frac", type=float, default=0.2)
    ap.add_argument("--concentration", type=float, default=50.0)
    ap.add_argument("--active-gene-frac", type=float, default=0.1)
    ap.add_argument("--active-gene-min-effect", type=float, default=0.2)
    ap.add_argument("--active-gene-max-effect", type=float, default=1.0)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--s-time", type=float, default=1.0)
    ap.add_argument("--s-guide", type=float, default=1.0)
    ap.add_argument("--s-tau", type=float, default=1.0)
    ap.add_argument(
        "--time-scale",
        default=None,
        help="Comma-separated per-interval scale (length days-1).",
    )
    ap.add_argument("--num-draws", type=int, default=200)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--skip-internal-fit", action="store_true")
    ap.add_argument("--write-anndata", action="store_true")
    ap.add_argument("--adata-out", default="data/sim_adata.h5ad")
    ap.add_argument("--guide-map-out", default="data/sim_guide_map.csv")
    ap.add_argument("--config-out", default=None)
    ap.add_argument("--metadata-out", default=None)
    ap.add_argument("--out-dir", default="out_fate_pipeline_sim")
    ap.add_argument("--run-export", action="store_true")
    ap.add_argument("--export-out", default=None)
    ap.add_argument("--export-guide-out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fate-key", default="lineages_fwd")
    ap.add_argument("--guide-key", default="guide")
    ap.add_argument("--covar-key", default="covar")
    ap.add_argument("--rep-key", default="rep")
    ap.add_argument("--day-key", default="day")
    args = ap.parse_args()

    if not (0.0 <= args.ntc_frac <= 1.0):
        raise SystemExit("--ntc-frac must be between 0 and 1")

    time_scale = _parse_time_scale(args.time_scale)
    if time_scale is not None:
        if any(v <= 0 for v in time_scale):
            raise SystemExit("--time-scale values must be positive")
        if args.days <= 1:
            if len(time_scale) != 0:
                raise SystemExit("--time-scale must be empty when --days <= 1")
        elif len(time_scale) != args.days - 1:
            raise SystemExit("--time-scale must have length days-1")

    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    fate_names = ("EC", "MES", "NEU")
    ref_fate = "EC"
    contrast_fate = "MES"

    model_args, params, gene_names, day_counts, sim_payload = _simulate_inputs(
        rng,
        N=args.cells,
        L=args.genes,
        G=args.guides,
        D=args.days,
        R=args.reps,
        Kmax=args.kmax,
        ntc_guides=args.ntc_guides,
        ntc_frac=args.ntc_frac,
        fate_names=fate_names,
        ref_fate=ref_fate,
        concentration=args.concentration,
        time_scale=time_scale,
        active_gene_frac=args.active_gene_frac,
        active_gene_min_effect=args.active_gene_min_effect,
        active_gene_max_effect=args.active_gene_max_effect,
    )

    true_betahat_daywise = _true_gene_daywise(
        params,
        L=args.genes,
        D=args.days,
        fate_names=fate_names,
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
    )

    if not args.skip_internal_fit:
        guide = fit_svi(
            *model_args,
            fate_names=fate_names,
            ref_fate=ref_fate,
            L=args.genes,
            G=args.guides,
            D=args.days,
            R=args.reps,
            Kmax=args.kmax,
            batch_size=args.batch_size,
            lr=args.lr,
            clip_norm=args.clip_norm,
            num_steps=args.num_steps,
            s_time=args.s_time,
            s_guide=args.s_guide,
            s_tau=args.s_tau,
            seed=args.seed,
        )

        summary = export_gene_summary_for_mash(
            guide=guide,
            model_args=model_args,
            gene_names=gene_names,
            fate_names=fate_names,
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            L=args.genes,
            D=args.days,
            num_draws=args.num_draws,
            out_csv=None,
        )

        betahat_cols = [c for c in summary.columns if c.startswith("betahat_d")]
        betahat_cols = sorted(betahat_cols, key=lambda c: int(c.split("d")[1]))
        if len(betahat_cols) != args.days:
            raise SystemExit("Internal fit summary missing betahat_d* columns.")
        est_betahat = summary[betahat_cols].to_numpy()
        metrics = _compute_metrics(est_betahat, true_betahat_daywise)
        _print_metrics("Internal fit (daywise)", metrics)

        if args.out_csv:
            out_path = Path(args.out_csv)
            _ensure_parent(out_path)
            _maybe_write(out_path, args.force)
            out = summary.copy()
            for d in range(args.days):
                out[f"true_betahat_d{d}"] = true_betahat_daywise[:, d]
                out[f"error_d{d}"] = (
                    out[f"betahat_d{d}"] - out[f"true_betahat_d{d}"]
                )
            out.to_csv(out_path, index=False)
            print("Wrote:", out_path)

    should_write = (
        args.write_anndata
        or args.run_export
        or args.config_out is not None
        or args.metadata_out is not None
    )
    if should_write:
        adata_path = Path(args.adata_out)
        guide_map_path = Path(args.guide_map_out)
        _ensure_parent(adata_path)
        _ensure_parent(guide_map_path)
        _maybe_write(adata_path, args.force)
        _maybe_write(guide_map_path, args.force)

        non_ntc_names, ntc_names = _guide_name_lists(args.guides, args.ntc_guides)
        guide_names = non_ntc_names + ntc_names

        _write_guide_map(
            guide_map_path,
            guide_names=guide_names,
            gene_names=gene_names,
            G=args.guides,
        )
        _write_anndata(
            adata_path,
            fate_key=args.fate_key,
            guide_key=args.guide_key,
            covar_key=args.covar_key,
            rep_key=args.rep_key,
            day_key=args.day_key,
            fate_names=fate_names,
            gene_names=gene_names,
            guide_names=guide_names,
            G=args.guides,
            sim_payload=sim_payload,
            true_betahat_daywise=true_betahat_daywise,
            seed=args.seed,
        )
        print("Wrote:", adata_path)
        print("Wrote:", guide_map_path)

        metadata_out = (
            Path(args.metadata_out)
            if args.metadata_out
            else Path(args.out_dir) / "sim_metadata.yaml"
        )
        _ensure_parent(metadata_out)
        _maybe_write(metadata_out, args.force)
        _write_sim_metadata(
            metadata_out,
            args=args,
            fate_names=fate_names,
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            day_counts=day_counts,
            sim_payload=sim_payload,
            time_scale=time_scale,
        )
        print("Wrote:", metadata_out)

    if args.run_export:
        config_out = Path(args.config_out) if args.config_out else Path(args.out_dir) / "sim_config.yaml"
        _ensure_parent(config_out)
        _maybe_write(config_out, args.force)
        _write_config(
            config_out,
            adata_path=args.adata_out,
            guide_map_csv=args.guide_map_out,
            out_dir=args.out_dir,
            fate_key=args.fate_key,
            guide_key=args.guide_key,
            covar_key=args.covar_key,
            rep_key=args.rep_key,
            day_key=args.day_key,
            fate_names=fate_names,
            ref_fate=ref_fate,
            contrast_fate=contrast_fate,
            Kmax=args.kmax,
            D=args.days,
            R=args.reps,
            batch_size=args.batch_size,
            lr=args.lr,
            clip_norm=args.clip_norm,
            num_steps=args.num_steps,
            s_alpha=1.0,
            s_rep=1.0,
            s_gamma=1.0,
            s_time=args.s_time,
            s_guide=args.s_guide,
            s_tau=args.s_tau,
            likelihood_weight=1.0,
            time_scale=time_scale,
            num_draws=args.num_draws,
            seed=args.seed,
        )
        print("Wrote:", config_out)

        export_out = (
            Path(args.export_out)
            if args.export_out
            else Path(args.out_dir) / "gene_daywise_for_mash.csv"
        )
        export_guide_out = (
            Path(args.export_guide_out)
            if args.export_guide_out
            else Path(args.out_dir) / "guide_daywise_for_mash.csv"
        )
        _ensure_parent(export_out)
        _ensure_parent(export_guide_out)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "fit_pyro_export.py"),
            "--config",
            str(config_out),
            "--adata",
            args.adata_out,
            "--guide-map",
            args.guide_map_out,
            "--out-gene",
            str(export_out),
            "--out-guide",
            str(export_guide_out),
        ]
        subprocess.run(cmd, check=True)
        print("Wrote:", export_out)
        print("Wrote:", export_guide_out)

        exported = pd.read_csv(export_out)
        betahat_cols = [c for c in exported.columns if c.startswith("betahat_d")]
        betahat_cols = sorted(betahat_cols, key=lambda c: int(c.split("d")[1]))
        if not betahat_cols:
            raise SystemExit("Exported mash summary missing betahat_d* columns.")
        if len(betahat_cols) != args.days:
            raise SystemExit("Exported mash summary missing some betahat_d* columns.")
        true_daywise = pd.DataFrame(
            true_betahat_daywise,
            index=gene_names,
            columns=[f"true_betahat_d{d}" for d in range(args.days)],
        )
        exported = exported.merge(
            true_daywise, left_on="gene", right_index=True, how="left"
        )
        true_cols = [f"true_betahat_d{d}" for d in range(args.days)]
        if exported[true_cols].isna().any().any():
            raise SystemExit("Missing true_betahat values for exported genes.")
        est_betahat = exported[betahat_cols].to_numpy()
        true_betahat = exported[true_cols].to_numpy()
        metrics = _compute_metrics(est_betahat, true_betahat)
        _print_metrics("Exported fit (daywise)", metrics)


if __name__ == "__main__":
    main()
