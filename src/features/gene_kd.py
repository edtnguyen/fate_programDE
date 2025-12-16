"""Utilities to build gene-level knockdown matrices from guide assignments."""

import logging
from typing import Mapping

import click
import pandas as pd


def build_gene_kd_matrix(
    guide_df: pd.DataFrame,
    guide_to_gene: Mapping[str, str],
) -> pd.DataFrame:
    """
    Aggregate guide-level assignments to gene-level knockdown indicators.

    Parameters
    ----------
    guide_df
        DataFrame of shape (cells, guides) with 0/1 or count entries.
    guide_to_gene
        Mapping from guide identifier to gene name. All guides in ``guide_df``
        must appear in the mapping.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (cells, genes) containing 0/1 indicators where 1
        means that any guide targeting the gene is present in the cell.
    """
    missing_guides = [g for g in guide_df.columns if g not in guide_to_gene]
    if missing_guides:
        missing_str = ", ".join(missing_guides)
        raise KeyError(f"Guides missing from guide_to_gene: {missing_str}")

    # Convert to binary indicators and rename columns to gene IDs.
    guide_indicators = (guide_df.astype(bool)).astype(int)
    guide_indicators.columns = pd.Index(
        [guide_to_gene[guide] for guide in guide_indicators.columns],
        name="gene",
    )

    # Collapse duplicate gene columns. For binary indicators, taking the max()
    # is equivalent to a logical OR, meaning the gene is marked as knocked
    # down if *any* of its guides are present.
    gene_kd_df = guide_indicators.T.groupby(level=0).max().T
    gene_kd_df = gene_kd_df.sort_index(axis=1)

    return gene_kd_df


@click.command()
@click.argument("guide_data_path", type=click.Path(exists=True))
@click.argument("guide_to_gene_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(guide_data_path, guide_to_gene_path, output_path):
    """
    Aggregate guide-level assignments to gene-level knockdown indicators.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading guide data from {guide_data_path}")
    guide_df = pd.read_csv(guide_data_path, index_col=0)

    logger.info(f"Loading guide-to-gene mapping from {guide_to_gene_path}")
    guide_to_gene = pd.read_csv(guide_to_gene_path, index_col=0, header=None).squeeze("columns").to_dict()

    logger.info("Building gene knockdown matrix")
    gene_kd_df = build_gene_kd_matrix(guide_df, guide_to_gene)

    logger.info(f"Saving gene knockdown matrix to {output_path}")
    gene_kd_df.to_csv(output_path)

    logger.info("Done.")


if __name__ == '__main__':
    main()
