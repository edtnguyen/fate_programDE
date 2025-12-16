"""Core regression helpers for fate probability analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_glm_for_gene_fate(
    y: np.ndarray,
    kd_vec: np.ndarray,
    covar_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Fit an OLS regression of logit(fate probability) on KD + covariates.

    Parameters
    ----------
    y
        1D array of logit-transformed fate probabilities for cells of a
        specific day.
    kd_vec
        1D binary knockdown vector for the gene, aligned to ``y``.
    covar_df
        Covariate DataFrame aligned to ``y``. May contain categorical columns.

    Returns
    -------
    tuple
        (beta_kd, t_kd, p_kd) corresponding to the KD coefficient, its t-value,
        and p-value from the fitted model.
    """
    intercept = pd.Series(
        np.ones(len(kd_vec)),
        index=covar_df.index,
        name="intercept",
    )
    kd_series = pd.Series(
        kd_vec,
        index=covar_df.index,
        name="KD",
    )

    # One-hot encode categorical covariates
    covar_df_encoded = pd.get_dummies(covar_df, drop_first=True, dtype=float)

    design = pd.concat(
        [intercept, kd_series, covar_df_encoded],
        axis=1,
    )

    model = sm.OLS(y, design)
    fit = model.fit()

    beta_kd = float(fit.params["KD"])
    t_kd = float(fit.tvalues["KD"])
    p_kd = float(fit.pvalues["KD"])

    return beta_kd, t_kd, p_kd
