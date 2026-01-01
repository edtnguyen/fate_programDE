#!/usr/bin/env python
"""Simulate known parameters and test recovery with the Pyro fate model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

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
    export_gene_summary_for_ash,
    fit_svi,
    resolve_fate_names,
)


def _center_k(k_raw: np.ndarray, day: np.ndarray) -> np.ndarray:
    k_centered = k_raw.astype(np.float32)
    for d in np.unique(day):
        idx = day == d
        if idx.any():
            k_centered[idx] = k_raw[idx] - k_raw[idx].mean()
    return k_centered.astype(np.float32)


def _make_guide_map(G: int, L: int) -> np.ndarray:
    gene_of_guide = np.zeros(G + 1, dtype=np.int64)
    gene_of_guide[1:] = (np.arange(1, G + 1) - 1) % L + 1
    return gene_of_guide


def _simulate_parameters(
    rng: np.random.Generator, L: int, G: int, D: int, R: int, fstar: int
) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    params["alpha"] = rng.normal(0.0, 0.2, size=(fstar, D)).astype(np.float32)
    params["b"] = rng.normal(0.0, 0.2, size=(fstar, R)).astype(np.float32)
    params["gamma"] = rng.normal(0.0, 0.1, size=(fstar,)).astype(np.float32)

    params["tau"] = np.full((fstar,), 0.3, dtype=np.float32)
    params["sigma_time"] = np.full((fstar,), 0.15, dtype=np.float32)
    params["sigma_guide"] = np.full((fstar,), 0.2, dtype=np.float32)

    params["z0"] = rng.normal(0.0, 1.0, size=(L, fstar)).astype(np.float32)
    if D > 1:
        params["eps"] = rng.normal(0.0, 1.0, size=(L, fstar, D - 1)).astype(
            np.float32
        )
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
    fate_names: tuple[str, ...],
    ref_fate: str,
    concentration: float,
) -> tuple[tuple[torch.Tensor, ...], dict[str, np.ndarray], list[str], list[int]]:
    fate_names, non_ref_fates, _, non_ref_indices = resolve_fate_names(
        fate_names, ref_fate=ref_fate
    )
    fstar = len(non_ref_fates)

    day = rng.integers(0, D, size=N, dtype=np.int64)
    rep = rng.integers(0, R, size=N, dtype=np.int64)

    max_k = min(Kmax, G)
    k_raw = rng.integers(1, max_k + 1, size=N, dtype=np.int64)
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)
    for i in range(N):
        g = rng.choice(G, size=k_raw[i], replace=False) + 1
        guide_ids[i, : k_raw[i]] = g
        mask[i, : k_raw[i]] = 1.0

    gene_of_guide = _make_guide_map(G, L)

    params = _simulate_parameters(rng, L=L, G=G, D=D, R=R, fstar=fstar)
    theta_core = construct_theta_core(
        tau=torch.tensor(params["tau"]),
        z0=torch.tensor(params["z0"]),
        sigma_time=torch.tensor(params["sigma_time"]),
        eps=torch.tensor(params["eps"]) if params["eps"] is not None else None,
        D=D,
    )
    theta = add_zero_gene_row(theta_core)

    delta_core = construct_delta_core(
        sigma_guide=torch.tensor(params["sigma_guide"]),
        u=torch.tensor(params["u"]),
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

    model_args = (p_t, day_t, rep_t, k_t, gids_t, mask_t, gene_of_guide_t)
    gene_names = [f"Gene{idx + 1}" for idx in range(L)]
    day_counts = np.bincount(day, minlength=D).tolist()
    return model_args, params, gene_names, day_counts


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
    _, non_ref_fates, _, _ = resolve_fate_names(fate_names, ref_fate=ref_fate)
    contrast_idx = non_ref_fates.index(contrast_fate)

    theta_core = construct_theta_core(
        tau=torch.tensor(params["tau"]),
        z0=torch.tensor(params["z0"]),
        sigma_time=torch.tensor(params["sigma_time"]),
        eps=torch.tensor(params["eps"]) if params["eps"] is not None else None,
        D=D,
    )
    theta = add_zero_gene_row(theta_core).detach().cpu().numpy()
    theta_gene = theta[1:, contrast_idx, :]

    weights = np.asarray(day_counts, dtype=np.float64)
    weights = weights / weights.sum()
    return (theta_gene * weights[None, :]).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--genes", type=int, default=30)
    ap.add_argument("--guides", type=int, default=90)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--concentration", type=float, default=50.0)
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--num-draws", type=int, default=200)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    fate_names = ("EC", "MES", "NEU")
    ref_fate = "EC"
    contrast_fate = "MES"

    model_args, params, gene_names, day_counts = _simulate_inputs(
        rng,
        N=args.cells,
        L=args.genes,
        G=args.guides,
        D=args.days,
        R=args.reps,
        Kmax=args.kmax,
        fate_names=fate_names,
        ref_fate=ref_fate,
        concentration=args.concentration,
    )

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
        seed=args.seed,
    )

    summary = export_gene_summary_for_ash(
        guide=guide,
        model_args=model_args,
        gene_names=gene_names,
        fate_names=fate_names,
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
        L=args.genes,
        D=args.days,
        num_draws=args.num_draws,
        day_cell_counts=day_counts,
        weights=None,
        out_csv=None,
    )

    true_betahat = _true_gene_summary(
        params,
        L=args.genes,
        D=args.days,
        fate_names=fate_names,
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
        day_counts=day_counts,
    )
    est_betahat = summary["betahat"].to_numpy()

    rmse = float(np.sqrt(np.mean((est_betahat - true_betahat) ** 2)))
    corr = float(np.corrcoef(est_betahat, true_betahat)[0, 1]) if args.genes > 1 else float("nan")
    sign_acc = float(np.mean(np.sign(est_betahat) == np.sign(true_betahat)))

    print("Recovery metrics")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pearson r: {corr:.4f}")
    print(f"  Sign agreement: {sign_acc:.3f}")

    if args.out_csv:
        out = summary.copy()
        out["true_betahat"] = true_betahat
        out["error"] = out["betahat"] - out["true_betahat"]
        out.to_csv(args.out_csv, index=False)
        print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
