"""Pipeline helpers for the MOI-aware Pyro fate model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch


def make_k_centered(cell_df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """
    Center guide counts within day.

    Returns
    -------
    np.ndarray
        Day-centered k values for each cell.
    """
    if "day" not in cell_df.columns:
        raise KeyError("cell_df must contain a 'day' column")

    k = mask.sum(axis=1).astype(np.float32)
    day = cell_df["day"].to_numpy()
    k_centered = k.copy()
    for d in np.unique(day):
        idx = day == d
        if idx.any():
            k_centered[idx] = k[idx] - k[idx].mean()
    return k_centered.astype(np.float32)


def to_torch(
    cell_df: pd.DataFrame,
    p: np.ndarray,
    guide_ids: np.ndarray,
    mask: np.ndarray,
    gene_of_guide: np.ndarray,
    guide_to_gene: np.ndarray,
    n_guides_per_gene: np.ndarray,
    device: str,
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
]:
    """
    Convert numpy/pandas inputs to torch tensors on the requested device.

    Returns
    -------
    tuple
        Torch tensors (p_t, day_t, rep_t, guide_ids_t, mask_t, gene_of_guide_t,
        guide_to_gene_t, n_guides_per_gene_t).
    """
    if "day" not in cell_df.columns or "rep" not in cell_df.columns:
        raise KeyError("cell_df must contain 'day' and 'rep' columns")

    import torch

    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    day_t = torch.tensor(cell_df["day"].to_numpy(), dtype=torch.long, device=device)
    rep_t = torch.tensor(cell_df["rep"].to_numpy(), dtype=torch.long, device=device)
    guide_ids_t = torch.tensor(guide_ids, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    gene_of_guide_t = torch.tensor(gene_of_guide, dtype=torch.long, device=device)
    guide_to_gene_t = torch.tensor(guide_to_gene, dtype=torch.long, device=device)
    n_guides_per_gene_t = torch.tensor(n_guides_per_gene, dtype=torch.long, device=device)
    return (
        p_t,
        day_t,
        rep_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    )
