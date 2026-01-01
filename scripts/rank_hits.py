#!/usr/bin/env python
"""Rank gene hits from ash output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ash", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dt = pd.read_csv(args.ash)

    lfsr_t = cfg["lfsr_thresh"]
    q_t = cfg["qvalue_thresh"]

    dt["hit"] = (dt["lfsr"] < lfsr_t) & (dt["qvalue"] < q_t)
    dt = dt.sort_values(["hit", "lfsr"], ascending=[False, True])

    dt.to_csv(args.out, index=False)
    print("Wrote:", args.out)
    if "postmean" in dt.columns:
        cols = ["gene", "postmean", "lfsr", "qvalue"]
    else:
        cols = ["gene", "betahat", "lfsr", "qvalue"]
    print(dt.loc[dt["hit"]].head(20)[cols])


if __name__ == "__main__":
    main()
