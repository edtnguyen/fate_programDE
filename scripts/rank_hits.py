#!/usr/bin/env python
"""Rank gene hits from mashr or ash output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.pyro_io import normalize_config  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mash", default=None)
    ap.add_argument("--ash", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = normalize_config(yaml.safe_load(open(args.config)))
    if bool(args.mash) == bool(args.ash):
        raise SystemExit("Provide exactly one of --mash or --ash.")

    if args.mash:
        dt = pd.read_csv(args.mash)
        lfsr_cols = [c for c in dt.columns if c.startswith("lfsr_d")]
        if not lfsr_cols:
            raise SystemExit("Mash output missing lfsr_d* columns.")
        if "lfsr_min" not in dt.columns:
            dt["lfsr_min"] = dt[lfsr_cols].min(axis=1)
        if "best_day" not in dt.columns:
            day_order = sorted(lfsr_cols, key=lambda c: int(c.split("d")[1]))
            day_idx = [int(c.split("d")[1]) for c in day_order]
            min_pos = dt[day_order].to_numpy().argmin(axis=1)
            dt["best_day"] = [day_idx[i] for i in min_pos]
        thresh = cfg.get("mash_lfsr_thresh", 0.05)
        dt["hit_anyday"] = dt["lfsr_min"] < thresh
        dt = dt.sort_values(["hit_anyday", "lfsr_min"], ascending=[False, True])
    else:
        dt = pd.read_csv(args.ash)
        lfsr_t = cfg["lfsr_thresh"]
        q_t = cfg["qvalue_thresh"]
        dt["hit"] = (dt["lfsr"] < lfsr_t) & (dt["qvalue"] < q_t)
        dt = dt.sort_values(["hit", "lfsr"], ascending=[False, True])

    dt.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    if args.mash:
        cols = ["gene", "lfsr_min", "best_day"]
        print(dt.loc[dt["hit_anyday"]].head(20)[cols])
    else:
        if "postmean" in dt.columns:
            cols = ["gene", "postmean", "lfsr", "qvalue"]
        else:
            cols = ["gene", "betahat", "lfsr", "qvalue"]
        print(dt.loc[dt["hit"]].head(20)[cols])


if __name__ == "__main__":
    main()
