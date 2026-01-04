#!/usr/bin/env python
"""Evaluate daywise hit calls against daywise simulated ground truth."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


LFSR_RE = re.compile(r"^(?:gene_)?lfsr_d(\d+)$")
POST_RE = re.compile(r"^(?:gene_)?postmean_d(\d+)$")
POSTSD_RE = re.compile(r"^(?:gene_)?postsd_d(\d+)$")
TRUE_RE = re.compile(r"^true_betahat_d(\d+)$")


def _extract_day_cols(columns: list[str], pattern: re.Pattern[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for col in columns:
        match = pattern.match(col)
        if match:
            out[int(match.group(1))] = col
    return out


def _confusion(pred: np.ndarray, true: np.ndarray) -> dict[str, int]:
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    try:  # prefer scipy if available
        from scipy.stats import norm  # type: ignore

        return norm.cdf(x)
    except Exception:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lfsr-thresh", type=float, default=0.05)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) if args.config else {}
    cfg = cfg or {}
    lfsr_thresh_day = float(cfg.get("mash_lfsr_thresh_day", cfg.get("mash_lfsr_thresh", args.lfsr_thresh)))
    effect_eps = float(cfg.get("effect_size_eps", 0.2))
    pmin = float(cfg.get("effect_prob_large_min", 0.90))
    correction = str(cfg.get("mash_anyday_correction", "none")).lower()
    thresh_any_raw = cfg.get("mash_lfsr_thresh_anyday", None)
    min_active_days_for_hit = int(cfg.get("min_active_days_for_hit_anyday", 1))

    hits = pd.read_csv(args.hits)
    truth = pd.read_csv(args.truth)

    lfsr_cols = _extract_day_cols(hits.columns.tolist(), LFSR_RE)
    post_cols = _extract_day_cols(hits.columns.tolist(), POST_RE)
    postsd_cols = _extract_day_cols(hits.columns.tolist(), POSTSD_RE)
    true_cols = _extract_day_cols(truth.columns.tolist(), TRUE_RE)

    if not lfsr_cols:
        raise SystemExit("hits file missing lfsr_d* columns.")
    if not post_cols or not postsd_cols:
        raise SystemExit("hits file missing postmean_d* or postsd_d* columns.")
    if not true_cols:
        raise SystemExit("truth file missing true_betahat_d* columns.")

    days = sorted(set(lfsr_cols) & set(post_cols) & set(postsd_cols) & set(true_cols))
    if not days:
        raise SystemExit("No overlapping day indices between hits and truth.")

    if thresh_any_raw in (None, "null", "None", ""):
        if correction == "bonferroni":
            thresh_any = lfsr_thresh_day / max(len(days), 1)
        else:
            thresh_any = lfsr_thresh_day
    else:
        thresh_any = float(thresh_any_raw)

    merged = hits.merge(truth, on="gene", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping genes between hits and truth.")

    lfsr_mat = merged[[lfsr_cols[d] for d in days]].to_numpy()
    post_mat = merged[[post_cols[d] for d in days]].to_numpy()
    postsd_mat = merged[[postsd_cols[d] for d in days]].to_numpy()
    postsd_mat = np.maximum(postsd_mat, 1e-8)

    z1 = (effect_eps - post_mat) / postsd_mat
    z0 = (-effect_eps - post_mat) / postsd_mat
    p_small = _norm_cdf(z1) - _norm_cdf(z0)
    p_large = np.clip(1.0 - p_small, 0.0, 1.0)

    pred_day = (lfsr_mat < lfsr_thresh_day) & (p_large >= pmin)
    true_mat = merged[[true_cols[d] for d in days]].to_numpy()
    true_day = np.abs(true_mat) >= effect_eps

    rows = []
    for idx, d in enumerate(days):
        counts = _confusion(pred_day[:, idx], true_day[:, idx])
        metrics = _metrics(counts)
        rows.append({"day": d, **counts, **metrics})

    pred_flat = pred_day.ravel()
    true_flat = true_day.ravel()
    overall_counts = _confusion(pred_flat, true_flat)
    overall_metrics = _metrics(overall_counts)
    rows.append({"day": "overall", **overall_counts, **overall_metrics})

    pred_any_day = (lfsr_mat < thresh_any) & (p_large >= pmin)
    n_any_active = pred_any_day.sum(axis=1)
    pred_any = n_any_active >= min_active_days_for_hit
    true_any = true_day.any(axis=1)
    any_counts = _confusion(pred_any, true_any)
    any_metrics = _metrics(any_counts)
    rows.append({"day": "any", **any_counts, **any_metrics})

    out_df = pd.DataFrame(rows)

    print(
        "Using lfsr_thresh_day={:.4f}, t_any={:.4f}, effect_eps={:.3f}, pmin={:.2f}".format(
            lfsr_thresh_day, thresh_any, effect_eps, pmin
        )
    )
    print(out_df.to_string(index=False))

    out_path = Path(args.out_csv) if args.out_csv else Path(args.hits).parent / "daywise_confusion.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)

    summary = {
        "n_hits": int(pred_any.sum()),
        "n_any_active_dist": {
            str(k): int(v) for k, v in pd.Series(n_any_active).value_counts().sort_index().to_dict().items()
        },
        "plarge_called": {
            "mean": float(np.mean(p_large[pred_any_day])) if pred_any_day.any() else float("nan"),
            "max": float(np.max(p_large[pred_any_day])) if pred_any_day.any() else float("nan"),
        },
    }
    summary_path = out_path.parent / "pred_call_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
