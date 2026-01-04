#!/usr/bin/env python
"""Summarize permutation-run hit counts for calibration."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_LFSR_RE = re.compile(r"^(?:gene_)?lfsr_d(\d+)$")
_POST_RE = re.compile(r"^(?:gene_)?postmean_d(\d+)$")
_POSTSD_RE = re.compile(r"^(?:gene_)?postsd_d(\d+)$")


def _active_day_count(series: pd.Series) -> pd.Series:
    def count_days(val: object) -> int:
        text = "" if val is None else str(val)
        if not text:
            return 0
        return len([x for x in text.split(",") if x])

    return series.apply(count_days)


def _max_abs_postmean(df: pd.DataFrame) -> pd.Series:
    if "max_abs_postmean" in df.columns:
        return df["max_abs_postmean"]
    post_cols = [c for c in df.columns if c.startswith("gene_postmean_d")] or [
        c for c in df.columns if c.startswith("postmean_d")
    ]
    if not post_cols:
        return pd.Series(np.nan, index=df.index)
    return df[post_cols].abs().max(axis=1)


def _extract_day_cols(columns: list[str], pattern: re.Pattern[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for col in columns:
        match = pattern.match(col)
        if match:
            out[int(match.group(1))] = col
    return out


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    try:  # prefer scipy if available
        from scipy.stats import norm  # type: ignore

        return norm.cdf(x)
    except Exception:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _summ_stats(values: pd.Series) -> dict[str, float]:
    vals = values.dropna().to_numpy()
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mash", default=None)
    ap.add_argument("--agg", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if bool(args.mash) == bool(args.agg):
        raise SystemExit("Provide exactly one of --mash or --agg.")
    df = pd.read_csv(args.mash or args.agg)
    if args.mash:
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) if args.config else {}
        cfg = cfg or {}
        lfsr_thresh_day = float(cfg.get("mash_lfsr_thresh_day", cfg.get("mash_lfsr_thresh", 0.05)))
        effect_eps = float(cfg.get("effect_size_eps", 0.2))
        pmin = float(cfg.get("effect_prob_large_min", 0.90))
        correction = str(cfg.get("mash_anyday_correction", "none")).lower()
        thresh_any_raw = cfg.get("mash_lfsr_thresh_anyday", None)
        min_active_days_for_hit = int(cfg.get("min_active_days_for_hit_anyday", 1))

        lfsr_cols = _extract_day_cols(df.columns.tolist(), _LFSR_RE)
        post_cols = _extract_day_cols(df.columns.tolist(), _POST_RE)
        postsd_cols = _extract_day_cols(df.columns.tolist(), _POSTSD_RE)
        if not lfsr_cols or not post_cols or not postsd_cols:
            raise SystemExit("Mash file missing lfsr_d*, postmean_d*, or postsd_d* columns.")
        days = sorted(set(lfsr_cols) & set(post_cols) & set(postsd_cols))
        if not days:
            raise SystemExit("No overlapping day indices in mash file.")

        if thresh_any_raw in (None, "null", "None", ""):
            if correction == "bonferroni":
                thresh_any = lfsr_thresh_day / max(len(days), 1)
            else:
                thresh_any = lfsr_thresh_day
        else:
            thresh_any = float(thresh_any_raw)

        lfsr_mat = df[[lfsr_cols[d] for d in days]].to_numpy()
        post_mat = df[[post_cols[d] for d in days]].to_numpy()
        postsd_mat = df[[postsd_cols[d] for d in days]].to_numpy()
        postsd_mat = np.maximum(postsd_mat, 1e-8)

        z1 = (effect_eps - post_mat) / postsd_mat
        z0 = (-effect_eps - post_mat) / postsd_mat
        p_small = _norm_cdf(z1) - _norm_cdf(z0)
        p_large = np.clip(1.0 - p_small, 0.0, 1.0)

        active_mat = (lfsr_mat < lfsr_thresh_day) & (p_large >= pmin)
        df["active_days"] = [
            ",".join(str(d) for d, flag in zip(days, row) if flag) for row in active_mat
        ]
        active_any = (lfsr_mat < thresh_any) & (p_large >= pmin)
        df["n_any_active"] = active_any.sum(axis=1)
        df["hit_anyday"] = df["n_any_active"] >= min_active_days_for_hit
        hits = df["hit_anyday"].astype(bool)
    else:
        if "hit_anyday" not in df.columns:
            raise SystemExit("Aggregated file missing hit_anyday.")
        hits = df["hit_anyday"].astype(bool)

    active_counts = _active_day_count(df.get("active_days", pd.Series("", index=df.index)))
    active_dist = active_counts.value_counts().sort_index().to_dict()
    hit_active_dist = active_counts[hits].value_counts().sort_index().to_dict()

    max_abs = _max_abs_postmean(df)
    hit_stats = _summ_stats(max_abs[hits])
    non_hit_stats = _summ_stats(max_abs[~hits])

    n_hits = int(hits.sum())
    suggestion = "ok"
    if n_hits > 0:
        suggestion = "increase mash_se_inflate or mash_se_floor; consider stricter mash_lfsr_thresh_anyday"

    summary = {
        "total_genes": int(df.shape[0]),
        "n_hits": n_hits,
        "active_day_counts": {str(k): int(v) for k, v in active_dist.items()},
        "hit_active_day_counts": {str(k): int(v) for k, v in hit_active_dist.items()},
        "max_abs_postmean": {"hit": hit_stats, "non_hit": non_hit_stats},
        "suggestion": suggestion,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
