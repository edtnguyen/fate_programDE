"""IO helpers for Pyro fate pipeline scripts."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_day_to_int(day_series: pd.Series, D: int) -> np.ndarray:
    """
    Convert day labels to integer codes 0..D-1.
    """
    vals = day_series.astype(str).values
    day_num = []
    for v in vals:
        digits = "".join([c for c in v if c.isdigit()])
        if digits == "":
            raise ValueError(f"Day value '{v}' has no digits to parse")
        day_num.append(int(digits))
    uniq = np.sort(np.unique(day_num))
    if len(uniq) != D:
        raise ValueError(f"Expected {D} unique days, got {uniq}")
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[x] for x in day_num], dtype=np.int64)


def parse_rep_to_int(rep_array: Iterable, R: int) -> np.ndarray:
    """
    Convert replicate labels to integer codes 0..R-1.
    """
    rep = np.asarray(rep_array).astype(str)
    uniq = np.sort(np.unique(rep))
    if len(uniq) != R:
        raise ValueError(f"Expected {R} unique reps, got {uniq}")
    mapping = {u: i for i, u in enumerate(uniq)}
    return np.array([mapping[x] for x in rep], dtype=np.int64)


def get_fate_probs(adata, key: str, fates: list[str]) -> np.ndarray:
    """
    Return p: [N,3] float32 in order fates (e.g., EC, MES, NEU).
    """
    obj = adata.obsm[key]
    if hasattr(obj, "columns"):
        missing = [f for f in fates if f not in obj.columns]
        if missing:
            raise ValueError(
                f"Missing fates {missing} in adata.obsm['{key}'].columns={list(obj.columns)}"
            )
        p = obj[fates].to_numpy(dtype=np.float32)
    else:
        p = np.asarray(obj, dtype=np.float32)
        if p.shape[1] != len(fates):
            raise ValueError(
                f"adata.obsm['{key}'] has shape {p.shape}, expected second dim={len(fates)}"
            )
    p = np.clip(p, 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def load_guide_map(path: str) -> pd.DataFrame:
    guide_map = pd.read_csv(path)
    required = {"guide_name", "gene_name", "is_ntc"}
    if not required.issubset(set(guide_map.columns)):
        raise ValueError(
            f"guide_map_csv must contain columns {required}, got {list(guide_map.columns)}"
        )
    return guide_map


def build_id_maps(
    guide_names: list[str],
    guide_map_df: pd.DataFrame,
) -> Tuple[dict[str, int], np.ndarray, list[str], int, int]:
    """
    Build guide and gene indexing maps.
    """
    gene_names = sorted(
        set(guide_map_df.loc[guide_map_df["is_ntc"] == 0, "gene_name"].tolist())
    )
    gene_to_id = {g: i + 1 for i, g in enumerate(gene_names)}
    L = len(gene_names)

    nonntc_guides = sorted(
        guide_map_df.loc[guide_map_df["is_ntc"] == 0, "guide_name"].tolist()
    )
    guide_to_id = {g: i + 1 for i, g in enumerate(nonntc_guides)}
    guide_to_id_ntc = {
        g: 0
        for g in guide_map_df.loc[guide_map_df["is_ntc"] == 1, "guide_name"].tolist()
    }

    guide_name_to_gid = {}
    guide_name_to_gid.update(guide_to_id)
    guide_name_to_gid.update(guide_to_id_ntc)

    G = len(nonntc_guides)

    gid_to_gene = np.zeros(G + 1, dtype=np.int64)
    gm_nonntc = guide_map_df.loc[
        guide_map_df["is_ntc"] == 0, ["guide_name", "gene_name"]
    ]
    for guide_name, gene_name in gm_nonntc.itertuples(index=False):
        gid = guide_name_to_gid[guide_name]
        gid_to_gene[gid] = gene_to_id[gene_name]

    return guide_name_to_gid, gid_to_gene, gene_names, L, G


def guides_to_padded_from_csr(
    Gmat_csr: sp.csr_matrix,
    colnames: list[str] | None,
    guide_name_to_gid: dict[str, int],
    Kmax: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert CSR guide matrix to padded guide_ids/mask.
    """
    N = Gmat_csr.shape[0]
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)

    indptr = Gmat_csr.indptr
    indices = Gmat_csr.indices
    data = Gmat_csr.data

    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        cols = indices[start:end]
        vals = data[start:end]
        if cols.size == 0:
            continue
        if colnames is None:
            raise ValueError(
                "Guide matrix has no column names; provide a DataFrame or set adata.uns['guide_names']."
            )
        gids = np.array(
            [guide_name_to_gid.get(colnames[c], None) for c in cols], dtype=object
        )
        keep = np.array([g is not None for g in gids], dtype=bool)
        cols = cols[keep]
        vals = vals[keep]
        gids = np.array([int(g) for g in gids[keep]], dtype=np.int64)
        if gids.size == 0:
            continue
        if gids.size > Kmax:
            topk = np.argpartition(-vals, Kmax - 1)[:Kmax]
            gids = gids[topk]

        m = gids.size
        guide_ids[i, :m] = gids
        mask[i, :m] = 1.0

    return guide_ids, mask


def load_adata_inputs(adata, cfg: dict, guide_map_csv: str):
    """
    Load and preprocess model inputs from AnnData and the guide map CSV.
    """
    day_int = parse_day_to_int(adata.obs[cfg["day_key"]], D=cfg["D"])

    covar = adata.obsm[cfg["covar_key"]]
    if hasattr(covar, "columns"):
        rep_raw = covar[cfg["rep_key"]].to_numpy()
    else:
        if covar.dtype.names and cfg["rep_key"] in covar.dtype.names:
            rep_raw = covar[cfg["rep_key"]]
        else:
            raise ValueError("rep_key not found in covariates")
    rep_int = parse_rep_to_int(rep_raw, R=cfg["R"])

    p = get_fate_probs(adata, key=cfg["fate_prob_key"], fates=cfg["fates"])

    guide_obsm = adata.obsm[cfg["guide_key"]]
    if hasattr(guide_obsm, "columns"):
        guide_names = list(guide_obsm.columns)
        Gmat = guide_obsm.to_numpy()
    else:
        guide_names = adata.uns.get("guide_names", None)
        if guide_names is None:
            raise ValueError("Missing guide names in adata.uns['guide_names']")
        Gmat = guide_obsm
    Gmat = sp.csr_matrix(Gmat)

    guide_map_df = load_guide_map(guide_map_csv)
    guide_name_to_gid, gid_to_gene, gene_names, L, G = build_id_maps(
        guide_names, guide_map_df
    )

    guide_ids, mask = guides_to_padded_from_csr(
        Gmat, guide_names, guide_name_to_gid, Kmax=cfg["Kmax"]
    )

    cell_df = pd.DataFrame({"day": day_int, "rep": rep_int})
    k = mask.sum(axis=1).astype(np.int64)
    keep = k <= cfg["Kmax"]
    if not keep.all():
        cell_df = cell_df.loc[keep].reset_index(drop=True)
        p = p[keep]
        guide_ids = guide_ids[keep]
        mask = mask[keep]

    day_counts = [
        int((cell_df["day"].to_numpy() == d).sum()) for d in range(cfg["D"])
    ]

    return cell_df, p, guide_ids, mask, gid_to_gene, gene_names, L, G, day_counts
