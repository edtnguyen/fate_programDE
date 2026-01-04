#!/usr/bin/env python
"""Permute guide assignments within day/rep/k strata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class BinRange:
    low: int
    high: int


def _parse_bins(spec: str | None) -> list[BinRange]:
    if spec is None or spec.strip() == "":
        spec = "1-2,3-5,6-10,11-20"
    bins = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise SystemExit(f"Invalid bin token '{token}' (expected 'a-b').")
        low_s, high_s = token.split("-", 1)
        bins.append(BinRange(int(low_s), int(high_s)))
    return bins


def _assign_bin(k: int, bins: list[BinRange]) -> str:
    if k <= 0:
        return "k0"
    for b in bins:
        if b.low <= k <= b.high:
            return f"{b.low}-{b.high}"
    return f"{bins[-1].high + 1}+"


def _extract_series(container: object, key: str) -> pd.Series:
    if isinstance(container, pd.DataFrame):
        return container[key]
    if hasattr(container, "dtype") and getattr(container.dtype, "names", None):
        if key in container.dtype.names:
            return pd.Series(container[key])
    raise SystemExit(f"Key '{key}' not found in covariates.")


def _to_numpy(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "to_numpy"):
        return matrix.to_numpy()
    return np.asarray(matrix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata-in", required=True)
    ap.add_argument("--adata-out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--guide-key", default=None)
    ap.add_argument("--day-key", default=None)
    ap.add_argument("--covar-key", default=None)
    ap.add_argument("--rep-key", default=None)
    ap.add_argument("--k-bin-mode", default=None, choices=["exact", "bins"])
    ap.add_argument("--k-bins", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    guide_key = args.guide_key or cfg.get("guide_key", "guide")
    day_key = args.day_key or cfg.get("day_key", "day")
    covar_key = args.covar_key or cfg.get("covar_key", "covar")
    rep_key = args.rep_key or cfg.get("rep_key", "rep")
    mode = args.k_bin_mode or cfg.get("perm_k_bin_mode", "bins")
    bins_spec = args.k_bins or cfg.get("perm_k_bins", None)
    seed = args.seed if args.seed is not None else int(cfg.get("perm_seed", 0))

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - runtime guard
        raise SystemExit(f"anndata is required to permute guides: {exc}")

    adata = ad.read_h5ad(args.adata_in)
    if guide_key not in adata.obsm:
        raise SystemExit(f"Guide key '{guide_key}' not found in adata.obsm.")
    if day_key not in adata.obs:
        raise SystemExit(f"Day key '{day_key}' not found in adata.obs.")
    if covar_key not in adata.obsm:
        raise SystemExit(f"Covariate key '{covar_key}' not found in adata.obsm.")

    guide_mat = adata.obsm[guide_key]
    guide_vals = _to_numpy(guide_mat)
    k_raw = guide_vals.sum(axis=1).astype(int)

    day = adata.obs[day_key].astype(str).to_numpy()
    covar = adata.obsm[covar_key]
    rep = _extract_series(covar, rep_key).astype(str).to_numpy()

    if mode == "exact":
        k_bin = np.array([f"k{k}" for k in k_raw], dtype=object)
    else:
        bins = _parse_bins(bins_spec)
        k_bin = np.array([_assign_bin(int(k), bins) for k in k_raw], dtype=object)

    strata = pd.DataFrame({"day": day, "rep": rep, "k_bin": k_bin})
    rng = np.random.default_rng(seed)
    permuted = np.zeros_like(guide_vals)
    for _, idx in strata.groupby(["day", "rep", "k_bin"]).indices.items():
        idx = np.asarray(idx)
        permuted[idx] = guide_vals[rng.permutation(idx)]

    if isinstance(guide_mat, pd.DataFrame):
        permuted_mat = pd.DataFrame(permuted, index=guide_mat.index, columns=guide_mat.columns)
    else:
        permuted_mat = permuted
    adata_perm = adata.copy()
    adata_perm.obsm[guide_key] = permuted_mat

    out_path = Path(args.adata_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_perm.write_h5ad(out_path)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
