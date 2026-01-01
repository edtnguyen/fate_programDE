"""Input preprocessing and aggregation helpers for fate regression."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _validate_arrays(
    adata,
    covar_names: Sequence[str],
    fate_names: Sequence[str],
    guide_to_gene: Mapping[str, str],
) -> None:
    """Lightweight validation of expected AnnData fields and shapes."""
    required_obsm = {"covar", "fateprob", "guide"}
    missing = [key for key in required_obsm if key not in adata.obsm]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"AnnData.obsm missing required keys: {missing_str}")

    if "day" not in adata.obs:
        raise KeyError("AnnData.obs must contain a 'day' column")

    covar = adata.obsm["covar"]
    fateprob = adata.obsm["fateprob"]
    guides = adata.obsm["guide"]

    if covar.shape[1] != len(covar_names):
        raise ValueError(
            f"covar columns {covar.shape[1]} do not match covar_names length "
            f"{len(covar_names)}"
        )

    if fateprob.shape[1] != len(fate_names):
        raise ValueError(
            f"fateprob columns {fateprob.shape[1]} do not match fate_names "
            f"length {len(fate_names)}"
        )

    if guides.shape[1] != len(list(guide_to_gene.keys())):
        raise ValueError(
            "Number of guide columns in adata.obsm['guide'] does not match "
            "guide_to_gene mapping"
        )


def _build_base_frames(
    adata,
    covar_names: Sequence[str],
    fate_names: Sequence[str],
    guide_to_gene: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Construct DataFrames from AnnData obsm/obs entries."""
    _validate_arrays(
        adata,
        covar_names=covar_names,
        fate_names=fate_names,
        guide_to_gene=guide_to_gene,
    )

    obs_index = pd.Index(adata.obs_names, name="cell")

    covar_data = adata.obsm["covar"]
    if isinstance(covar_data, pd.DataFrame):
        covar_df = covar_data
    else:
        covar_df = pd.DataFrame(
            covar_data,
            index=obs_index,
            columns=list(covar_names),
        )

    fate_df = pd.DataFrame(
        adata.obsm["fateprob"],
        index=obs_index,
        columns=list(fate_names),
    )
    guide_df = pd.DataFrame(
        adata.obsm["guide"],
        index=obs_index,
        columns=list(guide_to_gene.keys()),
    )
    day_series = adata.obs["day"]

    return covar_df, fate_df, guide_df, day_series


def aggregate_data_to_guide_level(
    adata,
    guide_to_gene: Mapping[str, str],
    covar_names: Sequence[str],
    fate_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate cell-level fates/covariates to guide-level means (pseudobulk).

    Each cell is reused for every guide it carries (high MOI friendly). The
    returned DataFrames share a MultiIndex (day, guide_idx) so that guides
    are aggregated per-day.
    """
    _validate_arrays(
        adata,
        covar_names=covar_names,
        fate_names=fate_names,
        guide_to_gene=guide_to_gene,
    )

    guide_mat = adata.obsm["guide"]
    fate_mat = adata.obsm["fateprob"]
    covar_mat = adata.obsm["covar"]
    day_array = adata.obs["day"].to_numpy()

    if isinstance(guide_mat, pd.DataFrame):
        guide_ids = list(guide_mat.columns)
        guide_values = guide_mat.to_numpy()
    else:
        guide_ids = list(guide_to_gene.keys())
        guide_values = guide_mat

    if isinstance(fate_mat, pd.DataFrame):
        fate_df = fate_mat.copy()
    else:
        fate_df = pd.DataFrame(
            fate_mat,
            index=pd.Index(adata.obs_names, name="cell"),
            columns=list(fate_names),
        )

    if isinstance(covar_mat, pd.DataFrame):
        covar_df = covar_mat.copy()
    else:
        covar_df = pd.DataFrame(
            covar_mat,
            index=pd.Index(adata.obs_names, name="cell"),
            columns=list(covar_names),
        )
    categorical_cols = list(
        covar_df.select_dtypes(include=["object", "category"]).columns
    )
    if categorical_cols:
        covar_df = pd.get_dummies(
            covar_df,
            columns=categorical_cols,
            drop_first=False,
            dtype=float,
        )
    covar_cols = list(covar_df.columns)

    if sp.issparse(guide_values):
        coo = guide_values.tocoo()
        cell_idx = coo.row
        guide_idx = coo.col
    else:
        cell_idx, guide_idx = np.nonzero(np.asarray(guide_values))

    if len(cell_idx) == 0:
        raise ValueError("No guide assignments found in adata.obsm['guide'].")

    day_for_pairs = day_array[cell_idx]

    fate_long = pd.DataFrame(
        fate_df.to_numpy()[cell_idx],
        columns=fate_df.columns,
    )
    fate_long["day"] = day_for_pairs
    fate_long["guide_idx"] = guide_idx
    fate_agg = fate_long.groupby(["day", "guide_idx"]).mean()

    covar_long = pd.DataFrame(
        covar_df.to_numpy()[cell_idx],
        columns=covar_cols,
    )
    covar_long["day"] = day_for_pairs
    covar_long["guide_idx"] = guide_idx
    covar_agg = covar_long.groupby(["day", "guide_idx"]).mean()

    counts = (
        pd.DataFrame({"day": day_for_pairs, "guide_idx": guide_idx})
        .groupby(["day", "guide_idx"])
        .size()
        .to_frame("n_cells")
    )
    counts["guide_id"] = [
        guide_ids[i] for i in counts.index.get_level_values("guide_idx")
    ]
    counts["target_gene"] = counts["guide_id"].map(guide_to_gene)

    return fate_agg, covar_agg, counts
