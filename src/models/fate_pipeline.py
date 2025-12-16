"""Pipeline to run per-gene fate regressions and visualize QQ plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata
import click
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from src.features.gene_kd import build_gene_kd_matrix
from src.features.transforms import logit
from src.models.glm_analysis import fit_glm_for_gene_fate
from src.visualization.qqplots import plot_qq_by_day_and_fate


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
    # Basic shape validation up front.
    _validate_arrays(
        adata,
        covar_names=covar_names,
        fate_names=fate_names,
        guide_to_gene=guide_to_gene,
    )

    # Pull raw matrices so we can preserve sparsity for guide lookups.
    guide_mat = adata.obsm["guide"]
    fate_mat = adata.obsm["fateprob"]
    covar_mat = adata.obsm["covar"]
    day_array = adata.obs["day"].to_numpy()

    # Guide ids: prefer provided column names, otherwise fall back to mapping order.
    if isinstance(guide_mat, pd.DataFrame):
        guide_ids = list(guide_mat.columns)
        guide_values = guide_mat.to_numpy()
    else:
        guide_ids = list(guide_to_gene.keys())
        guide_values = guide_mat

    # Fate probabilities
    if isinstance(fate_mat, pd.DataFrame):
        fate_df = fate_mat.copy()
    else:
        fate_df = pd.DataFrame(
            fate_mat,
            index=pd.Index(adata.obs_names, name="cell"),
            columns=list(fate_names),
        )

    # Covariates: encode categoricals so aggregation returns proportions.
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

    # Identify all (cell, guide) pairs (cell reuse).
    if sp.issparse(guide_values):
        coo = guide_values.tocoo()
        cell_idx = coo.row
        guide_idx = coo.col
    else:
        cell_idx, guide_idx = np.nonzero(np.asarray(guide_values))

    if len(cell_idx) == 0:
        raise ValueError("No guide assignments found in adata.obsm['guide'].")

    # Build long-form tables for aggregation.
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

        # Inner loop over genes for the current day
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
        # Slice MultiIndex by day -> index becomes guide_idx
        fate_day = fate_logit_agg.loc[(day_value,)]
        covar_day = covar_agg.loc[(day_value,)]
        meta_day = meta_df.loc[(day_value,)]

        if meta_day.empty:
            continue

        is_ntc = meta_day["target_gene"].isin(ntc_set)
        n_ntc = int(is_ntc.sum())
        if n_ntc == 0:
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


def run_fate_glm_pipeline(
    adata,
    guide_to_gene: Mapping[str, str],
    covar_names: Sequence[str],
    fate_names: Sequence[str] = ("ec", "mes", "neu"),
    non_targeting_genes: Iterable[str] = (),
    *,
    make_plots: bool = True,
    plot_save_dir: Path | None = None,
    show_plots: bool = True,
    use_guide_agg: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper to run regressions and generate QQ plots.

    Parameters
    ----------
    adata
        AnnData with guide, covariate, and fate probability matrices.
    guide_to_gene
        Mapping from guide identifier to gene name.
    covar_names
        List of covariate names aligning to ``adata.obsm['covar']`` columns.
    fate_names
        List of fate names. Defaults to ("ec", "mes", "neu").
    non_targeting_genes
        Iterable of gene identifiers used as negative controls.
    make_plots
        If True, generate QQ plots for each (day, fate).
    plot_save_dir
        Optional directory to write plots. Ignored if ``make_plots`` is False.
    show_plots
        Whether to call ``plt.show()``. Set to False for headless environments.
    use_guide_agg
        If True, use guide-level pseudobulk regressions; otherwise per-cell.
    """
    if use_guide_agg:
        results_df = run_day_gene_fate_regressions_guide_agg(
            adata=adata,
            guide_to_gene=guide_to_gene,
            covar_names=covar_names,
            fate_names=fate_names,
            non_targeting_genes=non_targeting_genes,
        )
    else:
        results_df = run_day_gene_fate_regressions(
            adata=adata,
            guide_to_gene=guide_to_gene,
            covar_names=covar_names,
            fate_names=fate_names,
            non_targeting_genes=non_targeting_genes,
        )

    if make_plots:
        plot_qq_by_day_and_fate(
            results_df,
            fate_names=fate_names,
            save_dir=plot_save_dir,
            show=show_plots,
        )

    return results_df


@click.command()
@click.argument("adata_path", type=click.Path(exists=True))
@click.argument("guide_to_gene_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--covariates", default=None, help="Comma-separated list of covariate names."
)
@click.option(
    "--fates", default="ec,mes,neu", help="Comma-separated list of fate names."
)
@click.option(
    "--ntc-genes",
    default=None,
    help="Comma-separated list of non-targeting control gene names.",
)
@click.option(
    "--make-plots/--no-make-plots", default=True, help="Generate and save QQ plots."
)
@click.option("--plot-save-dir", default=None, help="Directory to save plots.")
@click.option(
    "--show-plots/--no-show-plots", default=True, help="Show plots interactively."
)
@click.option(
    "--use-guide-agg/--no-use-guide-agg",
    default=False,
    help="Use guide-level pseudobulk regressions instead of per-cell.",
)
def main(
    adata_path,
    guide_to_gene_path,
    output_path,
    covariates,
    fates,
    ntc_genes,
    make_plots,
    plot_save_dir,
    show_plots,
    use_guide_agg,
):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading AnnData object from {adata_path}")
    adata = anndata.read_h5ad(adata_path)

    logger.info(f"Loading guide to gene mapping from {guide_to_gene_path}")
    guide_to_gene = (
        pd.read_csv(guide_to_gene_path, header=None, index_col=0)
        .squeeze("columns")
        .to_dict()
    )

    covariate_names = covariates.split(",") if covariates else []
    fate_names = fates.split(",")
    ntc_gene_names = ntc_genes.split(",") if ntc_genes else []

    logger.info("Running fate GLM pipeline")
    results_df = run_fate_glm_pipeline(
        adata,
        guide_to_gene,
        covariate_names,
        fate_names,
        ntc_gene_names,
        make_plots=make_plots,
        plot_save_dir=plot_save_dir,
        show_plots=show_plots,
        use_guide_agg=use_guide_agg,
    )

    logger.info(f"Saving results to {output_path}")
    results_df.to_csv(output_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
