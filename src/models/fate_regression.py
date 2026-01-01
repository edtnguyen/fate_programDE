"""Fate regression routines for KD effect estimation."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from src.features.fate_preprocessing import (
    _build_base_frames,
    aggregate_data_to_guide_level,
    build_gene_kd_matrix,
)
from src.features.transforms import logit
from src.models.glm_analysis import fit_glm_for_gene_fate


def run_day_gene_fate_regressions(
    adata,
    guide_to_gene: Mapping[str, str],
    covar_names: Sequence[str],
    fate_names: Sequence[str],
    non_targeting_genes: Iterable[str],
) -> pd.DataFrame:
    """
    Run OLS regressions of logit(fate probability) on KD + covariates.

    Parameters
    ----------
    adata
        AnnData object with required matrices in ``.obsm`` and day labels in
        ``.obs['day']``.
    guide_to_gene
        Mapping from guide identifier to gene name.
    covar_names
        Names for covariate columns corresponding to ``adata.obsm['covar']``.
    fate_names
        Names for fate columns corresponding to ``adata.obsm['fateprob']``.
    non_targeting_genes
        Iterable of gene names used as negative controls.

    Returns
    -------
    pd.DataFrame
        Regression results with columns: gene, fate, day, beta_kd, t_kd, p_kd,
        is_ntc, and p_adj_kd (Benjamini-Hochberg corrected p-value).
    """
    covar_df, fate_df, guide_df, day_series = _build_base_frames(
        adata,
        covar_names=covar_names,
        fate_names=fate_names,
        guide_to_gene=guide_to_gene,
    )

    gene_kd_df = build_gene_kd_matrix(guide_df, guide_to_gene)

    fate_logit_df = pd.DataFrame(
        logit(fate_df.values), index=fate_df.index, columns=fate_df.columns
    )

    all_results: list[dict] = []

    unique_days = sorted(pd.unique(day_series))
    for day_value in tqdm(unique_days, desc="Processing days"):
        mask = day_series == day_value

        covar_df_day = covar_df.loc[mask]
        fate_logit_df_day = fate_logit_df.loc[mask]
        gene_kd_df_day = gene_kd_df.loc[mask]

        for gene, kd_series in tqdm(
            gene_kd_df_day.items(), desc=f"Day {day_value} genes", leave=False
        ):
            kd_vec = kd_series.values.astype(np.float64)

            for fate in fate_names:
                y = fate_logit_df_day[fate].values
                beta_kd, t_kd, p_kd = fit_glm_for_gene_fate(
                    y=y,
                    kd_vec=kd_vec,
                    covar_df=covar_df_day,
                )

                all_results.append(
                    {
                        "gene": gene,
                        "fate": fate,
                        "day": int(day_value),
                        "beta_kd": beta_kd,
                        "t_kd": t_kd,
                        "p_kd": p_kd,
                    }
                )

    results_df = pd.DataFrame(all_results)
    results_df["is_ntc"] = results_df["gene"].isin(non_targeting_genes)

    if not results_df.empty:
        _, results_df["p_adj_kd"] = fdrcorrection(results_df["p_kd"])

    return results_df


def run_day_gene_fate_regressions_guide_agg(
    adata,
    guide_to_gene: Mapping[str, str],
    covar_names: Sequence[str],
    fate_names: Sequence[str],
    non_targeting_genes: Iterable[str],
) -> pd.DataFrame:
    """
    Run regressions on guide-level pseudobulk (cell-reuse) data.

    Each (day, guide) row represents the mean fate probabilities and mean
    covariates across all cells carrying that guide on that day. Tests compare
    target guides for one gene against non-targeting control guides.
    """
    fate_agg, covar_agg, meta_df = aggregate_data_to_guide_level(
        adata,
        guide_to_gene=guide_to_gene,
        covar_names=covar_names,
        fate_names=fate_names,
    )

    fate_logit_agg = pd.DataFrame(
        logit(fate_agg.values),
        index=fate_agg.index,
        columns=fate_agg.columns,
    )

    ntc_set = set(non_targeting_genes)
    all_results: list[dict] = []

    unique_days = sorted(fate_agg.index.get_level_values("day").unique())
    for day_value in tqdm(unique_days, desc="Processing days (guide agg)"):
        fate_day = fate_logit_agg.loc[(day_value,)]
        covar_day = covar_agg.loc[(day_value,)]
        meta_day = meta_df.loc[(day_value,)]

        if meta_day.empty:
            continue

        is_ntc = meta_day["target_gene"].isin(ntc_set)
        n_ntc = int(is_ntc.sum())
        if n_ntc == 0:
            print(f"No NC guides at {day_value}")
            continue

        unique_genes = meta_day.loc[~is_ntc, "target_gene"].dropna().unique()

        for gene in tqdm(
            unique_genes, desc=f"Day {day_value} genes (guide agg)", leave=False
        ):
            is_target = meta_day["target_gene"] == gene
            n_guides_gene = int(is_target.sum())
            if n_guides_gene == 0:
                continue

            subset_mask = is_target | is_ntc
            n_guides_ntc = int(is_ntc.sum())
            if n_guides_ntc == 0:
                continue

            kd_vec = is_target[subset_mask].astype(float).to_numpy()
            if kd_vec.sum() < 1 or (len(kd_vec) - kd_vec.sum()) < 1:
                continue

            y_subset = fate_day.loc[subset_mask]
            covar_subset = covar_day.loc[subset_mask]

            for fate in fate_names:
                y = y_subset[fate].to_numpy()
                beta_kd, t_kd, p_kd = fit_glm_for_gene_fate(
                    y=y,
                    kd_vec=kd_vec,
                    covar_df=covar_subset,
                )

                all_results.append(
                    {
                        "gene": gene,
                        "fate": fate,
                        "day": int(day_value),
                        "beta_kd": beta_kd,
                        "t_kd": t_kd,
                        "p_kd": p_kd,
                        "n_guides_gene": n_guides_gene,
                        "n_guides_ntc": n_guides_ntc,
                        "n_cells_gene": int(meta_day.loc[is_target, "n_cells"].sum()),
                        "n_cells_ntc": int(meta_day.loc[is_ntc, "n_cells"].sum()),
                    }
                )

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df["is_ntc"] = False
        _, results_df["p_adj_kd"] = fdrcorrection(results_df["p_kd"])

    return results_df
