#!/usr/bin/env python
"""Rank gene hits from mashr or ash output."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.pyro_io import normalize_config  # noqa: E402

_POSTMEAN_RE = re.compile(r"^(?:gene_)?postmean_d(\d+)$")
_POSTSD_RE = re.compile(r"^(?:gene_)?postsd_d(\d+)$")
_LFSR_RE = re.compile(r"^(?:gene_)?lfsr_d(\d+)$")
_DAY_RE = re.compile(r"d(\d+)$")


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


def _join_days(mask: np.ndarray, days: list[int]) -> list[str]:
    out: list[str] = []
    for row in mask:
        out.append(",".join(str(d) for d, flag in zip(days, row) if flag))
    return out


def _day_key(col: str) -> int:
    match = _DAY_RE.search(col)
    if match:
        return int(match.group(1))
    return -1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mash", "--in-mash-gene", dest="mash", default=None)
    ap.add_argument("--ash", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = normalize_config(yaml.safe_load(open(args.config)))
    if bool(args.mash) == bool(args.ash):
        raise SystemExit("Provide exactly one of --mash or --ash.")

    if args.mash:
        dt = pd.read_csv(args.mash)
        lfsr_cols = _extract_day_cols(dt.columns.tolist(), _LFSR_RE)
        post_cols = _extract_day_cols(dt.columns.tolist(), _POSTMEAN_RE)
        postsd_cols = _extract_day_cols(dt.columns.tolist(), _POSTSD_RE)
        if not lfsr_cols:
            raise SystemExit("Mash output missing lfsr_d* columns.")
        if not post_cols or not postsd_cols:
            raise SystemExit("Mash output missing postmean_d* or postsd_d* columns.")

        days = sorted(set(lfsr_cols) & set(post_cols) & set(postsd_cols))
        if not days:
            raise SystemExit("No overlapping day indices across lfsr/postmean/postsd columns.")

        lfsr_mat = dt[[lfsr_cols[d] for d in days]].to_numpy()
        post_mat = dt[[post_cols[d] for d in days]].to_numpy()
        postsd_mat = dt[[postsd_cols[d] for d in days]].to_numpy()
        postsd_mat = np.maximum(postsd_mat, 1e-8)

        lfsr_thresh_day = float(cfg.get("mash_lfsr_thresh_day", cfg.get("mash_lfsr_thresh", 0.05)))
        effect_eps = float(cfg.get("effect_size_eps", 0.0))
        pmin = float(cfg.get("effect_prob_large_min", 0.90))
        correction = str(cfg.get("mash_anyday_correction", "none")).lower()
        thresh_any = cfg.get("mash_lfsr_thresh_anyday", None)
        if thresh_any in (None, "null", "None", ""):
            if correction == "bonferroni":
                thresh_any = lfsr_thresh_day / max(len(days), 1)
            else:
                thresh_any = lfsr_thresh_day
        thresh_any = float(thresh_any)
        min_active_days_for_hit = int(cfg.get("min_active_days_for_hit_anyday", 1))

        z1 = (effect_eps - post_mat) / postsd_mat
        z0 = (-effect_eps - post_mat) / postsd_mat
        p_small = _norm_cdf(z1) - _norm_cdf(z0)
        p_large = np.clip(1.0 - p_small, 0.0, 1.0)

        for idx, d in enumerate(days):
            dt[f"plarge_d{d}"] = p_large[:, idx]

        active_mat = (lfsr_mat < lfsr_thresh_day) & (p_large >= pmin)
        dt["active_days"] = _join_days(active_mat, days)
        dt["n_active_days"] = active_mat.sum(axis=1)

        best_idx = lfsr_mat.argmin(axis=1)
        dt["best_day"] = [days[i] for i in best_idx]

        active_lfsr = np.where(active_mat, lfsr_mat, np.inf)
        min_active_lfsr = active_lfsr.min(axis=1)
        dt["min_lfsr_among_active"] = min_active_lfsr
        best_active_idx = active_lfsr.argmin(axis=1)
        dt["best_day_among_active"] = [
            (days[i] if np.isfinite(min_active_lfsr[row]) else np.nan)
            for row, i in enumerate(best_active_idx)
        ]

        dt["max_abs_postmean"] = np.abs(post_mat).max(axis=1)

        active_any_mat = (lfsr_mat < thresh_any) & (p_large >= pmin)
        dt["active_days_any"] = _join_days(active_any_mat, days)
        dt["n_any_active"] = active_any_mat.sum(axis=1)
        dt["hit_anyday"] = dt["n_any_active"] >= min_active_days_for_hit

        sort_cols = ["hit_anyday", "min_lfsr_among_active", "max_abs_postmean"]
        dt = dt.sort_values(sort_cols, ascending=[False, True, False])
    else:
        dt = pd.read_csv(args.ash)
        lfsr_t = cfg["lfsr_thresh"]
        q_t = cfg["qvalue_thresh"]
        dt["hit"] = (dt["lfsr"] < lfsr_t) & (dt["qvalue"] < q_t)
        dt = dt.sort_values(["hit", "lfsr"], ascending=[False, True])

    dt.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    if args.mash:
        cols = [
            "gene",
            "hit_anyday",
            "active_days",
            "n_active_days",
            "active_days_any",
            "n_any_active",
            "best_day_among_active",
            "min_lfsr_among_active",
            "max_abs_postmean",
        ]
        extra = [
            c
            for c in dt.columns
            if _POSTMEAN_RE.match(c)
            or _POSTSD_RE.match(c)
            or _LFSR_RE.match(c)
            or c.startswith("plarge_d")
        ]
        extra = sorted(extra, key=_day_key)
        cols = [c for c in cols if c in dt.columns] + extra
        print(dt.loc[dt["hit_anyday"]].head(20)[cols])
    else:
        if "postmean" in dt.columns:
            cols = ["gene", "postmean", "lfsr", "qvalue"]
        else:
            cols = ["gene", "betahat", "lfsr", "qvalue"]
        print(dt.loc[dt["hit"]].head(20)[cols])


if __name__ == "__main__":
    main()
