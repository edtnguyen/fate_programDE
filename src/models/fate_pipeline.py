"""Pipeline to run per-gene fate regressions and visualize QQ plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata
import click
import pandas as pd

from src.models.fate_preprocessing import aggregate_data_to_guide_level
from src.models.fate_regression import (
    run_day_gene_fate_regressions,
    run_day_gene_fate_regressions_guide_agg,
)
from src.visualization.qqplots import plot_qq_by_day_and_fate


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
