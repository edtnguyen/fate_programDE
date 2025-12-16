"""Pipeline to run per-gene fate regressions and visualize QQ plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata
import click
import numpy as np
import pandas as pd
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
        is_ntc.
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
        for gene, kd_series in tqdm(gene_kd_df_day.items(), desc=f"Day {day_value} genes", leave=False):
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
    """
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
@click.option("--covariates", default=None, help="Comma-separated list of covariate names.")
@click.option("--fates", default="ec,mes,neu", help="Comma-separated list of fate names.")
@click.option("--ntc-genes", default=None, help="Comma-separated list of non-targeting control gene names.")
@click.option("--make-plots/--no-make-plots", default=True, help="Generate and save QQ plots.")
@click.option("--plot-save-dir", default=None, help="Directory to save plots.")
@click.option("--show-plots/--no-show-plots", default=True, help="Show plots interactively.")
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
):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading AnnData object from {adata_path}")
    adata = anndata.read_h5ad(adata_path)

    logger.info(f"Loading guide to gene mapping from {guide_to_gene_path}")
    guide_to_gene = pd.read_csv(guide_to_gene_path, header=None, index_col=0).squeeze("columns").to_dict()

    covariate_names = covariates.split(',') if covariates else []
    fate_names = fates.split(',')
    ntc_gene_names = ntc_genes.split(',') if ntc_genes else []

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
    )

    logger.info(f"Saving results to {output_path}")
    results_df.to_csv(output_path)

    logger.info("Done.")


if __name__ == '__main__':
    main()
