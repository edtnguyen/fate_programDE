#!/usr/bin/env python3
"""
Generate simulated data and run the pyro model to test its logic.
"""
import torch
import numpy as np
import pandas as pd
from src.models import pyro_model


def generate_mock_inputs(
    num_cells=100,
    num_guides=10,
    num_genes=5,
    num_days=4,
    num_reps=2,
    max_guides_per_cell=3,
):
    """
    Generate mock inputs for the pyro model.
    """
    fate_names = ["EC", "MES", "NEU"]
    ref_fate = "EC"
    L = num_genes
    G = num_guides
    D = num_days
    R = num_reps
    Kmax = max_guides_per_cell

    # Generate fate probabilities
    p = np.random.rand(num_cells, len(fate_names))
    p /= p.sum(axis=1, keepdims=True)
    p_t = torch.tensor(p, dtype=torch.float32)

    # Generate day, rep, and k (number of guides)
    day_t = torch.randint(0, D, (num_cells,))
    rep_t = torch.randint(0, R, (num_cells,))
    k_t = torch.randint(1, Kmax + 1, (num_cells,))

    # Generate guide_ids and mask
    guide_ids = np.zeros((num_cells, Kmax), dtype=int)
    mask = np.zeros((num_cells, Kmax), dtype=bool)
    for i in range(num_cells):
        num_guides_for_cell = k_t[i].item()
        guides = np.random.choice(G, num_guides_for_cell, replace=False) + 1
        guide_ids[i, :num_guides_for_cell] = guides
        mask[i, :num_guides_for_cell] = True

    guide_ids_t = torch.tensor(guide_ids, dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.float32)

    # Generate gene_of_guide
    # Last guide is NTC, maps to gene 0 (baseline)
    gene_of_guide = (np.arange(G + 1) % L) + 1
    gene_of_guide[0] = 0  # baseline guide
    gene_of_guide_t = torch.tensor(gene_of_guide, dtype=torch.long)
    guide_to_gene = gene_of_guide[1:] - 1
    n_guides_per_gene = np.bincount(guide_to_gene, minlength=L).astype(np.int64)
    guide_to_gene_t = torch.tensor(guide_to_gene, dtype=torch.long)
    n_guides_per_gene_t = torch.tensor(n_guides_per_gene, dtype=torch.long)

    model_args = (
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    )

    fit_svi_args = {
        "L": L,
        "G": G,
        "D": D,
        "R": R,
        "Kmax": Kmax,
        "fate_names": fate_names,
        "ref_fate": ref_fate,
        "batch_size": 32,
        "lr": 0.01,
        "clip_norm": 10.0,
        "num_steps": 2, # Small number for testing
    }

    return model_args, fit_svi_args


def main():
    """
    Main function to generate data and run the model.
    """
    print("Generating mock data...")
    model_args, fit_svi_args = generate_mock_inputs()
    print("Mock data generated.")
    print("Running fit_svi...")
    guide = pyro_model.fit_svi(*model_args, **fit_svi_args)
    print("fit_svi finished successfully.")
    print("Guide object:", guide)


if __name__ == "__main__":
    main()
