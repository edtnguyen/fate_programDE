#!/usr/bin/env python
"""Compare mashr modes based on aggregated gene hits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODE_RE = re.compile(r"gene_from_guide_mash_(.+)\\.csv$")


def _infer_mode(path: str) -> str:
    name = Path(path).name
    match = MODE_RE.search(name)
    return match.group(1) if match else name


def _load_hits(path: str) -> dict:
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise SystemExit(f"{path} missing 'gene' column")
    if "hit_anyday" in df.columns:
        hits = df.loc[df["hit_anyday"]].copy()
    else:
        lfsr_col = _score_column(df)
        hits = df.loc[df[lfsr_col] < 0.05].copy()
    return {
        "df": df,
        "hits": hits,
    }


def _score_column(df: pd.DataFrame) -> str:
    if "min_lfsr_among_active" in df.columns:
        return "min_lfsr_among_active"
    if "min_lfsr" in df.columns:
        return "min_lfsr"
    if "lfsr_min" in df.columns:
        return "lfsr_min"
    return "gene"


def _top_genes(df: pd.DataFrame, k: int = 50) -> set[str]:
    score_col = _score_column(df)
    if score_col in df.columns and score_col != "gene":
        ranked = df.sort_values(score_col, ascending=True)
    else:
        ranked = df.copy()
    return set(ranked["gene"].head(k).tolist())


def _active_day_counts(df: pd.DataFrame) -> np.ndarray:
    if "active_days" not in df.columns:
        return np.array([], dtype=int)
    counts = []
    for entry in df["active_days"].fillna(""):
        entry = str(entry)
        if not entry:
            counts.append(0)
        else:
            counts.append(len([x for x in entry.split(",") if x]))
    return np.asarray(counts, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mode_data = {}
    for path in args.inputs:
        mode = _infer_mode(path)
        payload = _load_hits(path)
        df = payload["df"]
        hits = payload["hits"]
        score_col = _score_column(df)
        active_counts = _active_day_counts(hits)
        mode_data[mode] = {
            "hits": set(hits["gene"].tolist()),
            "n_hits": int(hits.shape[0]),
            "median_score": float(df[score_col].median()) if score_col in df.columns else float("nan"),
            "median_active_days": float(np.median(active_counts)) if active_counts.size else float("nan"),
            "top50": _top_genes(df),
        }

    modes = sorted(mode_data)
    rows = []
    for i, mode_a in enumerate(modes):
        for mode_b in modes[i + 1 :]:
            hits_a = mode_data[mode_a]["hits"]
            hits_b = mode_data[mode_b]["hits"]
            union = hits_a | hits_b
            inter = hits_a & hits_b
            jaccard = float(len(inter) / len(union)) if union else 0.0
            top_a = mode_data[mode_a]["top50"]
            top_b = mode_data[mode_b]["top50"]
            top_overlap = int(len(top_a & top_b))
            rows.append(
                {
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "hits_a": mode_data[mode_a]["n_hits"],
                    "hits_b": mode_data[mode_b]["n_hits"],
                    "median_score_a": mode_data[mode_a]["median_score"],
                    "median_score_b": mode_data[mode_b]["median_score"],
                    "median_active_days_a": mode_data[mode_a]["median_active_days"],
                    "median_active_days_b": mode_data[mode_b]["median_active_days"],
                    "jaccard_hits": jaccard,
                    "top50_overlap": top_overlap,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
