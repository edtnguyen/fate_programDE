#!/usr/bin/env python
"""Render the Pyro fate model graph to an image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pyro  # noqa: E402
import torch  # noqa: E402

from scripts.pyro_io import build_id_maps, load_guide_map, normalize_config  # noqa: E402
from src.models.pyro_model import fate_model  # noqa: E402


def _dummy_inputs(
    cfg: dict,
    guide_map_csv: str,
    n_cells: int,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    guide_map_df = load_guide_map(guide_map_csv)
    guide_names = guide_map_df["guide_name"].tolist()
    _, gene_of_guide, _, L, G = build_id_maps(guide_names, guide_map_df)

    D = cfg["D"]
    R = cfg["R"]
    Kmax = cfg["Kmax"]
    fate_names = tuple(cfg["fates"])

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(len(fate_names)), size=n_cells).astype(np.float32)
    day = np.arange(n_cells, dtype=np.int64) % D
    rep = np.arange(n_cells, dtype=np.int64) % R

    guide_ids = np.zeros((n_cells, Kmax), dtype=np.int64)
    mask = np.zeros((n_cells, Kmax), dtype=np.float32)
    max_guides = min(Kmax, max(1, min(G, 3)))
    gids = (np.arange(max_guides) % max(G, 1)) + 1
    for i in range(n_cells):
        guide_ids[i, :max_guides] = gids
        mask[i, :max_guides] = 1.0

    k = mask.sum(axis=1).astype(np.float32)

    model_args = (
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(day, dtype=torch.long),
        torch.tensor(rep, dtype=torch.long),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(guide_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(gene_of_guide, dtype=torch.long),
    )

    model_kwargs = {
        "fate_names": fate_names,
        "ref_fate": cfg.get("ref_fate", "EC"),
        "L": L,
        "G": G,
        "D": D,
        "R": R,
        "Kmax": Kmax,
        "s_alpha": cfg.get("s_alpha", 1.0),
        "s_rep": cfg.get("s_rep", 1.0),
        "s_gamma": cfg.get("s_gamma", 1.0),
        "s_tau": cfg.get("s_tau", 1.0),
        "s_time": cfg.get("s_time", 1.0),
        "s_guide": cfg.get("s_guide", 1.0),
        "time_scale": cfg.get("time_scale", None),
    }

    return model_args, model_kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--guide-map", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-cells", type=int, default=6)
    ap.add_argument("--render-distributions", action="store_true")
    ap.add_argument("--render-params", action="store_true")
    ap.add_argument("--render-deterministic", action="store_true")
    args = ap.parse_args()

    cfg = normalize_config(yaml.safe_load(open(args.config)))
    guide_map_csv = args.guide_map or cfg.get("guide_map_csv", None)
    if guide_map_csv is None:
        raise SystemExit("guide_map_csv must be set in config or passed via --guide-map")

    out_dir = Path(cfg.get("out_dir", "out_fate_pipeline"))
    out_path = Path(args.out) if args.out else out_dir / "pyro_model.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pyro.set_rng_seed(0)
    model_args, model_kwargs = _dummy_inputs(cfg, guide_map_csv, args.n_cells)

    pyro.render_model(
        fate_model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        filename=str(out_path),
        render_distributions=args.render_distributions,
        render_params=args.render_params,
        render_deterministic=args.render_deterministic,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
