#!/usr/bin/env python
"""Sweep s_time and s_guide with multi-seed averaging for recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch  # noqa: E402
import pyro  # noqa: E402

from scripts import simulate_recovery as sim  # noqa: E402
from src.models.pyro_model import export_gene_summary_for_ash, fit_svi  # noqa: E402


def _parse_list(value: str, cast=float) -> list:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-data", type=int, default=0)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--s-time-values", default="1.0,0.5")
    ap.add_argument("--s-guide-values", default="1.0,0.5")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--genes", type=int, default=30)
    ap.add_argument("--guides", type=int, default=90)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--ntc-guides", type=int, default=1)
    ap.add_argument("--ntc-frac", type=float, default=0.2)
    ap.add_argument("--concentration", type=float, default=5.0)
    ap.add_argument("--num-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--num-draws", type=int, default=200)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-detail", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = _parse_list(args.seeds, int)
    s_time_values = _parse_list(args.s_time_values, float)
    s_guide_values = _parse_list(args.s_guide_values, float)

    rng = np.random.default_rng(args.seed_data)

    fate_names = ("EC", "MES", "NEU")
    ref_fate = "EC"
    contrast_fate = "MES"

    model_args, params, gene_names, day_counts, _ = sim._simulate_inputs(
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
    )

    true_betahat = sim._true_gene_summary(
        params,
        L=args.genes,
        D=args.days,
        fate_names=fate_names,
        ref_fate=ref_fate,
        contrast_fate=contrast_fate,
        day_counts=day_counts,
    )

    detail_rows = []
    summary_rows = []

    for s_time in s_time_values:
        for s_guide in s_guide_values:
            seed_metrics = []
            betahat_list = []
            for seed in seeds:
                pyro.set_rng_seed(seed)
                torch.manual_seed(seed)

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
                    s_time=float(s_time),
                    s_guide=float(s_guide),
                    seed=seed,
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

                est_betahat = summary["betahat"].to_numpy()
                metrics = sim._compute_metrics(est_betahat, true_betahat)
                seed_metrics.append(metrics)
                betahat_list.append(est_betahat)

                detail_rows.append(
                    {
                        "s_time": float(s_time),
                        "s_guide": float(s_guide),
                        "seed": int(seed),
                        **metrics,
                    }
                )

            betahat_avg = np.mean(np.stack(betahat_list, axis=0), axis=0)
            avg_metrics = sim._compute_metrics(betahat_avg, true_betahat)

            rmse_mean = float(np.mean([m["rmse"] for m in seed_metrics]))
            corr_mean = float(np.mean([m["corr"] for m in seed_metrics]))
            sign_mean = float(np.mean([m["sign_acc"] for m in seed_metrics]))

            summary_rows.append(
                {
                    "s_time": float(s_time),
                    "s_guide": float(s_guide),
                    "rmse_mean": rmse_mean,
                    "corr_mean": corr_mean,
                    "sign_mean": sign_mean,
                    "rmse_avg": avg_metrics["rmse"],
                    "corr_avg": avg_metrics["corr"],
                    "sign_avg": avg_metrics["sign_acc"],
                    "num_seeds": len(seeds),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    out_summary = Path(args.out_summary)
    out_detail = Path(args.out_detail)
    _ensure_parent(out_summary)
    _ensure_parent(out_detail)
    _maybe_write(out_summary, args.force)
    _maybe_write(out_detail, args.force)

    summary_df.to_csv(out_summary, index=False)
    detail_df.to_csv(out_detail, index=False)

    print("Prior sweep summary")
    for row in summary_rows:
        print(
            f"s_time={row['s_time']}, s_guide={row['s_guide']}: "
            f"rmse_mean={row['rmse_mean']:.4f}, r_mean={row['corr_mean']:.3f}, "
            f"sign_mean={row['sign_mean']:.3f}, rmse_avg={row['rmse_avg']:.4f}, "
            f"r_avg={row['corr_avg']:.3f}, sign_avg={row['sign_avg']:.3f}"
        )
    print("Wrote:", out_summary)
    print("Wrote:", out_detail)


if __name__ == "__main__":
    main()
