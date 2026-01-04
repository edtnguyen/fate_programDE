#!/usr/bin/env python
"""Aggregate guide-level mashr output to gene-level hits."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DAY_RE = re.compile(r"^([a-z_]+)_d(\d+)$")


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _extract_day_cols(columns: list[str], prefix: str) -> dict[int, str]:
    out: dict[int, str] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for col in columns:
        match = pattern.match(col)
        if match:
            out[int(match.group(1))] = col
    return out


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return float(value)


def _normal_lfsr(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.clip(sd, 1e-8, None)
    z = (0.0 - mu) / sd
    erf_vec = np.vectorize(math.erf)
    cdf = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))
    return np.minimum(cdf, 1.0 - cdf)


def _aggregate_meta_fixed(dt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    postmean_cols = _extract_day_cols(dt.columns.tolist(), "postmean_d")
    postsd_cols = _extract_day_cols(dt.columns.tolist(), "postsd_d")
    lfsr_cols = _extract_day_cols(dt.columns.tolist(), "lfsr_d")

    if not postmean_cols or not postsd_cols or not lfsr_cols:
        raise SystemExit("Input must include postmean_d*, postsd_d*, and lfsr_d* columns.")

    days = sorted(set(postmean_cols) & set(postsd_cols))
    if set(days) != set(lfsr_cols):
        missing = sorted(set(postmean_cols) ^ set(postsd_cols) ^ set(lfsr_cols))
        raise SystemExit(f"Missing day columns for indices: {missing}")

    postmean_cols_list = [postmean_cols[d] for d in days]
    postsd_cols_list = [postsd_cols[d] for d in days]
    lfsr_cols_list = [lfsr_cols[d] for d in days]

    effect_eps = float(cfg.get("effect_size_eps", 0.0))
    lfsr_thresh_day = float(cfg.get("mash_lfsr_thresh_day", cfg.get("mash_lfsr_thresh", 0.05)))
    correction = str(cfg.get("mash_anyday_correction", "none")).lower()
    thresh_any = _as_optional_float(cfg.get("mash_lfsr_thresh_anyday", None))
    if correction == "bonferroni":
        if thresh_any is None:
            thresh_any = lfsr_thresh_day / max(len(days), 1)
    else:
        thresh_any = lfsr_thresh_day

    rows = []
    for gene, group in dt.groupby("gene"):
        mu = group[postmean_cols_list].to_numpy(dtype=float)
        sd = group[postsd_cols_list].to_numpy(dtype=float)
        sd = np.clip(sd, 1e-8, None)
        weights = 1.0 / (sd**2)
        sumw = weights.sum(axis=0)
        sumw_safe = np.where(sumw > 0, sumw, np.nan)
        mu_gene = (weights * mu).sum(axis=0) / sumw_safe
        sd_gene = np.sqrt(1.0 / sumw_safe)

        gene_lfsr = _normal_lfsr(mu_gene, sd_gene)
        hit_day = (gene_lfsr < lfsr_thresh_day) & (np.abs(mu_gene) >= effect_eps)
        hit_anyday = bool(np.any((gene_lfsr < thresh_any) & (np.abs(mu_gene) >= effect_eps)))

        active_days = ",".join([f"d{d}" for d, hit in zip(days, hit_day) if hit])
        best_day = int(days[int(np.nanargmin(gene_lfsr))])
        if np.any(hit_day):
            active_idx = np.where(hit_day)[0]
            best_active_idx = active_idx[int(np.nanargmin(gene_lfsr[active_idx]))]
            best_day_among_active = int(days[best_active_idx])
            min_lfsr_among_active = float(np.nanmin(gene_lfsr[active_idx]))
        else:
            best_day_among_active = np.nan
            min_lfsr_among_active = np.nan

        guide_lfsr_min = group[lfsr_cols_list].min(axis=1)
        max_abs_postmean = float(np.nanmax(np.abs(mu_gene)))

        row: dict[str, object] = {
            "gene": gene,
            "n_guides": int(group.shape[0]),
            "best_day": best_day,
            "best_day_among_active": best_day_among_active,
            "active_days": active_days,
            "hit_anyday": hit_anyday,
            "max_abs_postmean": max_abs_postmean,
            "min_lfsr": float(np.nanmin(gene_lfsr)),
            "min_lfsr_among_active": min_lfsr_among_active,
            "guide_min_lfsr_min": float(guide_lfsr_min.min()),
            "median_guide_lfsr_min": float(guide_lfsr_min.median()),
        }

        for idx, day in enumerate(days):
            row[f"gene_postmean_d{day}"] = float(mu_gene[idx])
            row[f"gene_postsd_d{day}"] = float(sd_gene[idx])
            row[f"gene_lfsr_d{day}"] = float(gene_lfsr[idx])

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-mash-guide", dest="input_csv", default=None)
    ap.add_argument("--out-gene", dest="output_csv", default=None)
    ap.add_argument("--in", dest="input_csv_legacy", default=None)
    ap.add_argument("--out", dest="output_csv_legacy", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default=None)
    args = ap.parse_args()

    input_csv = args.input_csv or args.input_csv_legacy
    output_csv = args.output_csv or args.output_csv_legacy
    if input_csv is None or output_csv is None:
        raise SystemExit("Provide --in-mash-guide/--out-gene (or legacy --in/--out).")

    cfg = _load_config(args.config)
    method = args.method or cfg.get("guide_to_gene_agg_method", "meta_fixed")
    if method != "meta_fixed":
        raise SystemExit(f"Aggregation method '{method}' not implemented.")

    dt = pd.read_csv(input_csv)
    if "gene" not in dt.columns:
        raise SystemExit("Input missing required 'gene' column.")

    out_df = _aggregate_meta_fixed(dt, cfg)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
