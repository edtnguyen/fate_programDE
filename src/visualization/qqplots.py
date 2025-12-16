"""QQ plot utilities for KD regression results."""

import logging
from pathlib import Path
from typing import Iterable, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def qq_plot_against_uniform(
    p_values: Iterable[float],
    ax: plt.Axes,
    label: str,
) -> None:
    """Draw a QQ plot of p-values against the Uniform(0, 1) expectation."""
    p = np.asarray(list(p_values))
    p = p[np.isfinite(p)]
    if p.size == 0:
        return

    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    m = p.size
    p_sorted = np.sort(p)
    exp = (np.arange(1, m + 1) - 0.5) / m

    ax.plot(
        -np.log10(exp),
        -np.log10(p_sorted),
        marker=".",
        linestyle="none",
        label=label,
    )


def plot_targeting_vs_ntc(
    results_df: pd.DataFrame,
    fate: str,
    day_value: int,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot QQ curves for targeting vs non-targeting genes for one day × fate.

    Parameters
    ----------
    results_df
        DataFrame containing columns: gene, fate, day, p_kd, is_ntc.
    fate
        Fate label to plot.
    day_value
        Day to subset.
    ax
        Optional Matplotlib axes.

    Returns
    -------
    plt.Axes
        The axes used for plotting.
    """
    df_day = results_df[(results_df["day"] == day_value) & (results_df["fate"] == fate)]
    if df_day.empty:
        raise ValueError(f"No results for fate={fate}, day={day_value}")

    if ax is None:
        _, ax = plt.subplots()

    p_target = df_day.loc[~df_day["is_ntc"], "p_kd"].values
    p_ntc = df_day.loc[df_day["is_ntc"], "p_kd"].values

    qq_plot_against_uniform(
        p_target,
        ax=ax,
        label="Targeting genes",
    )
    qq_plot_against_uniform(
        p_ntc,
        ax=ax,
        label="Non-targeting genes",
    )

    max_x = ax.get_xlim()[1]
    ax.plot(
        [0, max_x],
        [0, max_x],
        linestyle="--",
        color="gray",
    )

    ax.set_xlabel("Expected -log10(p) under Uniform(0,1)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(f"QQ plot: fate = {fate}, day = {day_value}")
    ax.legend()
    return ax


def plot_qq_by_day_and_fate(
    results_df: pd.DataFrame,
    fate_names: Iterable[str],
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Generate QQ plots for each (day, fate) pair.

    Parameters
    ----------
    results_df
        Regression results with columns: gene, fate, day, p_kd, is_ntc.
    fate_names
        List of fate names to generate plots for.
    save_dir
        Optional directory to save figures as PNGs. Created if missing.
    show
        Whether to display figures via ``plt.show()``.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    unique_days = sorted(results_df["day"].unique())

    for day_value in unique_days:
        for fate in fate_names:
            ax = plot_targeting_vs_ntc(
                results_df,
                fate=fate,
                day_value=day_value,
            )
            plt.tight_layout()

            if save_dir is not None:
                filename = save_dir / f"qq_fate-{fate}_day-{day_value}.png"
                ax.figure.savefig(filename, dpi=200)

            if show:
                plt.show()
            else:
                plt.close(ax.figure)

@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--fates", "-f", multiple=True, help="Fates to plot (can be specified multiple times).")
@click.option("--save-dir", "-s", type=click.Path(), help="Directory to save plots.")
@click.option("--show/--no-show", default=True, help="Show plots interactively.")
def main(results_path, fates, save_dir, show):
    """
    Generate QQ plots from regression results.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading results from {results_path}")
    results_df = pd.read_csv(results_path)

    if not fates:
        fates = results_df["fate"].unique()
        logger.info(f"No fates specified, using all found fates: {fates}")

    logger.info(f"Generating QQ plots for fates: {fates}")
    plot_qq_by_day_and_fate(results_df, fates, save_dir=save_dir, show=show)

    logger.info("Done.")


if __name__ == "__main__":
    main()

