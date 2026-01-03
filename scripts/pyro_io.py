"""IO helpers for Pyro fate pipeline scripts."""

from __future__ import annotations

from typing import Iterable, Tuple

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_INT_CFG_KEYS = {
    "Kmax",
    "D",
    "R",
    "batch_size",
    "num_steps",
    "num_posterior_draws",
    "seed",
    "bootstrap_reps",
    "bootstrap_num_steps",
    "bootstrap_num_draws",
    "bootstrap_seed",
    "diagnostics_num_steps",
    "diagnostics_num_draws",
    "diagnostics_perm_steps",
    "diagnostics_perm_draws",
    "diagnostics_seed",
    "sanity_num_draws",
}
_FLOAT_CFG_KEYS = {
    "lr",
    "clip_norm",
    "s_alpha",
    "s_rep",
    "s_gamma",
    "s_time",
    "s_guide",
    "s_tau",
    "likelihood_weight",
    "bootstrap_frac",
    "lfsr_thresh",
    "qvalue_thresh",
    "mash_lfsr_thresh",
    "diagnostics_holdout_frac",
    "sanity_min_abs_effect",
}


def _coerce_int(cfg: dict, key: str) -> None:
    if key not in cfg:
        return
    val = cfg[key]
    if val is None:
        return
    if isinstance(val, bool):
        raise ValueError(f"Config '{key}' must be an int, got bool.")
    if isinstance(val, (np.integer, int)):
        cfg[key] = int(val)
        return
    if isinstance(val, (np.floating, float)):
        if not float(val).is_integer():
            raise ValueError(f"Config '{key}' must be an int, got {val}.")
        cfg[key] = int(val)
        return
    if isinstance(val, str):
        try:
            val_f = float(val)
        except ValueError as exc:
            raise ValueError(f"Config '{key}' must be an int, got '{val}'.") from exc
        if not val_f.is_integer():
            raise ValueError(f"Config '{key}' must be an int, got '{val}'.")
        cfg[key] = int(val_f)
        return
    raise ValueError(f"Config '{key}' must be an int, got {type(val)}.")


def _coerce_float(cfg: dict, key: str) -> None:
    if key not in cfg:
        return
    val = cfg[key]
    if val is None:
        return
    if isinstance(val, bool):
        raise ValueError(f"Config '{key}' must be a float, got bool.")
    if isinstance(val, (np.floating, float, np.integer, int)):
        cfg[key] = float(val)
        return
    if isinstance(val, str):
        try:
            cfg[key] = float(val)
        except ValueError as exc:
            raise ValueError(f"Config '{key}' must be a float, got '{val}'.") from exc
        return
    raise ValueError(f"Config '{key}' must be a float, got {type(val)}.")


def normalize_config(cfg: dict) -> dict:
    """
    Coerce numeric config fields loaded from YAML into proper types.
    """
    cfg = dict(cfg)
    for key in _INT_CFG_KEYS:
        _coerce_int(cfg, key)
    for key in _FLOAT_CFG_KEYS:
        _coerce_float(cfg, key)

    weights = cfg.get("weights", None)
    if weights is not None:
        if isinstance(weights, (list, tuple, np.ndarray)):
            cfg["weights"] = [float(w) for w in weights]
        else:
            raise ValueError("Config 'weights' must be a list of floats or null.")

    time_scale = cfg.get("time_scale", None)
    if time_scale is not None:
        if isinstance(time_scale, (list, tuple, np.ndarray)):
            cfg["time_scale"] = [float(v) for v in time_scale]
        else:
            raise ValueError("Config 'time_scale' must be a list of floats or null.")
        if any(v <= 0 for v in cfg["time_scale"]):
            raise ValueError("Config 'time_scale' entries must be positive.")
        D = cfg.get("D", None)
        if D is not None:
            if D <= 1:
                if len(cfg["time_scale"]) != 0:
                    raise ValueError("Config 'time_scale' must be empty when D <= 1.")
            elif len(cfg["time_scale"]) != D - 1:
                raise ValueError(f"Config 'time_scale' must have length D-1={D - 1}.")

    return cfg


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
    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        cols = indices[start:end]
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
        gids = np.array([int(g) for g in gids[keep]], dtype=np.int64)
        if gids.size == 0:
            continue
        if gids.size > Kmax:
            raise ValueError(
                f"Cell {i} has {gids.size} guides after mapping; exceeds Kmax={Kmax}."
            )

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

    k_raw = np.asarray(Gmat.getnnz(axis=1)).astype(np.int64).ravel()
    keep = k_raw <= cfg["Kmax"]
    if not keep.all():
        dropped = int((~keep).sum())
        logger.info("Dropping %d cells with k > Kmax (%d).", dropped, cfg["Kmax"])
        p = p[keep]
        day_int = day_int[keep]
        rep_int = rep_int[keep]
        Gmat = Gmat[keep]

    guide_map_df = load_guide_map(guide_map_csv)
    missing_guides = sorted(set(guide_names) - set(guide_map_df["guide_name"]))
    if missing_guides:
        sample = ", ".join(missing_guides[:5])
        logger.warning(
            "Guide map missing %d guide names (e.g., %s). These will be ignored.",
            len(missing_guides),
            sample,
        )
    guide_name_to_gid, gid_to_gene, gene_names, L, G = build_id_maps(
        guide_names, guide_map_df
    )

    guide_ids, mask = guides_to_padded_from_csr(
        Gmat, guide_names, guide_name_to_gid, Kmax=cfg["Kmax"]
    )

    k = mask.sum(axis=1).astype(np.int64)
    if not np.all(k <= cfg["Kmax"]):
        raise ValueError("Found cells with k > Kmax after filtering; check preprocessing.")

    cell_df = pd.DataFrame({"day": day_int, "rep": rep_int})

    day_counts = [
        int((cell_df["day"].to_numpy() == d).sum()) for d in range(cfg["D"])
    ]

    return cell_df, p, guide_ids, mask, gid_to_gene, gene_names, L, G, day_counts
