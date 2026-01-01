"""Pipeline stubs for the MOI-aware Pyro fate model and ash workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch


@dataclass
class Config:
    Kmax: int = 20
    D: int = 4
    R: int = 2
    L: int = 300
    G: int = 2000
    device: str = "cpu"
    batch_size: int = 8192
    lr: float = 1e-3
    clip_norm: float = 5.0
    num_steps_stage1: int = 1000
    num_steps_stage2: int = 5000
    num_draws: int = 1000
    weights: Optional[List[float]] = None
    eps_effect: float = 0.1
    lfsr_thresh: float = 0.05
    qvalue_thresh: float = 0.10
    out_dir: str = "out_fate_pipeline"
    gene_summary_csv: str = "gene_summary_for_ash.csv"
    ash_out_csv: str = "gene_summary_ash_out.csv"
    run_ash_r: str = "run_ash.R"


def load_inputs() -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[List[int]],
    pd.DataFrame,
    List[str],
]:
    """
    Load raw inputs for the Pyro fate model pipeline.

    Returns
    -------
    tuple
        (cell_df, p, guides_per_cell, guide_map, gene_names).
    """
    raise NotImplementedError("Input loading stub; implementation pending.")


def build_padded_guides(
    guides_per_cell: List[List[int]],
    guide_map: pd.DataFrame,
    Kmax: int,
    G: int,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build padded guide ID and mask matrices plus gene-of-guide mapping.

    Returns
    -------
    tuple
        (guide_ids, mask, gene_of_guide) arrays.
    """
    raw_to_gid = dict(zip(guide_map["raw_guide_id"], guide_map["guide_id"]))
    gid_to_gene = np.zeros(G + 1, dtype=np.int64)
    gid_to_gene[0] = 0
    for _, row in guide_map.iterrows():
        gid = int(row["guide_id"])
        gene = int(row["gene_id"])
        if gid >= 0:
            gid_to_gene[gid] = gene

    N = len(guides_per_cell)
    guide_ids = np.zeros((N, Kmax), dtype=np.int64)
    mask = np.zeros((N, Kmax), dtype=np.float32)

    for i, raw_list in enumerate(guides_per_cell):
        gids: List[int] = []
        for raw in raw_list:
            if raw in raw_to_gid:
                gids.append(int(raw_to_gid[raw]))
        gids = gids[:Kmax]
        if gids:
            guide_ids[i, : len(gids)] = np.array(gids, dtype=np.int64)
            mask[i, : len(gids)] = 1.0

    return guide_ids, mask, gid_to_gene


def filter_cells_by_k(
    cell_df: pd.DataFrame,
    p: np.ndarray,
    guide_ids: np.ndarray,
    mask: np.ndarray,
    Kmax: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter cells to those with guide count k <= Kmax.

    Returns
    -------
    tuple
        Filtered (cell_df, p, guide_ids, mask).
    """
    if guide_ids.shape != mask.shape:
        raise ValueError("guide_ids and mask must have the same shape")
    if p.shape[0] != guide_ids.shape[0] or len(cell_df) != guide_ids.shape[0]:
        raise ValueError("cell_df, p, and guide_ids must have the same length")

    k = mask.sum(axis=1).astype(np.int64)
    keep = k <= Kmax
    cell_df2 = cell_df.loc[keep].reset_index(drop=True)
    return cell_df2, p[keep], guide_ids[keep], mask[keep]


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
    device: str,
) -> Tuple[
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
        Torch tensors (p_t, day_t, rep_t, guide_ids_t, mask_t, gene_of_guide_t).
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
    return p_t, day_t, rep_t, guide_ids_t, mask_t, gene_of_guide_t


def run_ash_rscript(in_csv: str, out_csv: str, r_script_path: str) -> None:
    """
    Run the ash R script on the gene summary CSV.

    Returns
    -------
    None
    """
    raise NotImplementedError("ash runner stub; implementation pending.")


def main(cfg: Config) -> None:
    """
    Orchestrate the full Pyro + ash pipeline.

    Returns
    -------
    None
    """
    raise NotImplementedError("Pipeline stub; implementation pending.")
