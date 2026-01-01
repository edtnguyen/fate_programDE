#!/usr/bin/env python3
"""Generate small mock AnnData + guide map for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as exc:  # pragma: no cover - runtime guard
    raise SystemExit(f"anndata is required to run this script: {exc}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata-out", default="data/adata.h5ad")
    ap.add_argument("--guide-map-out", default="data/guide_map.csv")
    ap.add_argument("--fate-key", default="lineages_fwd")
    ap.add_argument("--guide-key", default="guide")
    ap.add_argument("--covar-key", default="covar")
    ap.add_argument("--rep-key", default="rep")
    ap.add_argument("--day-key", default="day")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells-per-combo", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    adata_path = Path(args.adata_out)
    guide_map_path = Path(args.guide_map_out)
    _ensure_parent(adata_path)
    _ensure_parent(guide_map_path)
    _maybe_write(adata_path, args.force)
    _maybe_write(guide_map_path, args.force)

    rng = np.random.default_rng(args.seed)

    days = ["d0", "d1", "d2", "d3"]
    reps = ["r0", "r1"]
    n_cells = len(days) * len(reps) * args.cells_per_combo

    cell_ids = [f"cell{i}" for i in range(n_cells)]
    day_values = []
    rep_values = []
    for d in days:
        for r in reps:
            for _ in range(args.cells_per_combo):
                day_values.append(d)
                rep_values.append(r)

    obs = pd.DataFrame({args.day_key: day_values}, index=cell_ids)
    covar = pd.DataFrame({args.rep_key: rep_values}, index=cell_ids)

    fate_probs = rng.uniform(0.1, 1.0, size=(n_cells, 3))
    fate_probs = fate_probs / fate_probs.sum(axis=1, keepdims=True)
    fate_df = pd.DataFrame(
        fate_probs, index=cell_ids, columns=["EC", "MES", "NEU"]
    )

    guide_names = ["g1", "g2", "g3", "ntc1"]
    guide = np.zeros((n_cells, len(guide_names)), dtype=np.int64)
    for i in range(n_cells):
        n_guides = rng.integers(1, 4)
        gids = rng.choice([0, 1, 2], size=n_guides, replace=False)
        guide[i, gids] = 1
        if rng.random() < 0.3:
            guide[i, 3] = 1
    guide_df = pd.DataFrame(guide, index=cell_ids, columns=guide_names)

    adata = ad.AnnData(X=np.zeros((n_cells, 1)), obs=obs)
    adata.obsm[args.covar_key] = covar
    adata.obsm[args.fate_key] = fate_df
    adata.obsm[args.guide_key] = guide_df

    guide_map = pd.DataFrame(
        {
            "guide_name": guide_names,
            "gene_name": ["GeneA", "GeneB", "GeneC", "NTC"],
            "is_ntc": [0, 0, 0, 1],
        }
    )

    adata.write_h5ad(adata_path)
    guide_map.to_csv(guide_map_path, index=False)
    print(f"Wrote {adata_path}")
    print(f"Wrote {guide_map_path}")


if __name__ == "__main__":
    main()
