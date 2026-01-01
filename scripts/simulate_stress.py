#!/usr/bin/env python
"""Run stress tests for recovery on noisier simulated data."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.pyro_io import normalize_config  # noqa: E402

DEFAULT_SCENARIOS = [
    {"label": "C10_N200_S300", "concentration": 10.0, "cells": 200, "steps": 300},
    {"label": "C5_N200_S300", "concentration": 5.0, "cells": 200, "steps": 300},
    {"label": "C10_N100_S200", "concentration": 10.0, "cells": 100, "steps": 200},
    {"label": "C5_N100_S200", "concentration": 5.0, "cells": 100, "steps": 200},
    {"label": "C2_N100_S200", "concentration": 2.0, "cells": 100, "steps": 200},
]
DEFAULT_SEEDS = [0, 1]

DEFAULT_GENES = 30
DEFAULT_GUIDES = 90
DEFAULT_KMAX = 4
DEFAULT_NUM_DRAWS = 100
DEFAULT_BATCH = 128
DEFAULT_LR = 1e-3
DEFAULT_CLIP = 5.0

DEFAULT_WARN_CORR = 0.6
DEFAULT_WARN_SIGN = 0.7

RMSE_RE = re.compile(r"RMSE:\s*([0-9.]+)")
CORR_RE = re.compile(r"Pearson r:\s*([0-9.\-]+)")
SIGN_RE = re.compile(r"Sign agreement:\s*([0-9.]+)")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite.")


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return normalize_config(cfg)


def _parse_metrics(stdout: str) -> dict[str, float]:
    rmse = RMSE_RE.search(stdout)
    corr = CORR_RE.search(stdout)
    sign = SIGN_RE.search(stdout)
    if not (rmse and corr and sign):
        raise ValueError("Could not parse recovery metrics from output")
    return {
        "rmse": float(rmse.group(1)),
        "corr": float(corr.group(1)),
        "sign": float(sign.group(1)),
    }


def _run_scenario(
    *,
    python_exe: str,
    script_path: Path,
    env: dict[str, str],
    seed: int,
    cells: int,
    steps: int,
    concentration: float,
    genes: int,
    guides: int,
    kmax: int,
    num_draws: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
) -> dict[str, float]:
    cmd = [
        python_exe,
        str(script_path),
        "--seed",
        str(seed),
        "--cells",
        str(cells),
        "--num-steps",
        str(steps),
        "--concentration",
        str(concentration),
        "--genes",
        str(genes),
        "--guides",
        str(guides),
        "--kmax",
        str(kmax),
        "--num-draws",
        str(num_draws),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--clip-norm",
        str(clip_norm),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if run.returncode != 0:
        raise RuntimeError(f"simulate_recovery failed:\n{run.stdout}\n{run.stderr}")
    return _parse_metrics(run.stdout)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-detail", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    scenarios = cfg.get("stress_scenarios", DEFAULT_SCENARIOS)
    seeds = cfg.get("stress_seeds", DEFAULT_SEEDS)

    genes = cfg.get("stress_genes", DEFAULT_GENES)
    guides = cfg.get("stress_guides", DEFAULT_GUIDES)
    kmax = cfg.get("stress_kmax", DEFAULT_KMAX)
    num_draws = cfg.get("stress_num_draws", DEFAULT_NUM_DRAWS)
    batch_size = cfg.get("stress_batch_size", DEFAULT_BATCH)
    lr = cfg.get("stress_lr", DEFAULT_LR)
    clip_norm = cfg.get("stress_clip_norm", DEFAULT_CLIP)

    warn_corr = cfg.get("stress_warn_corr", DEFAULT_WARN_CORR)
    warn_sign = cfg.get("stress_warn_sign", DEFAULT_WARN_SIGN)

    python_exe = sys.executable
    script_path = ROOT / "scripts" / "simulate_recovery.py"

    tmpdir = os.environ.get("TMPDIR", str(ROOT / ".tmp"))
    mpldir = os.environ.get("MPLCONFIGDIR", str(ROOT / ".tmp" / "mpl"))
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(mpldir, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = tmpdir
    env["MPLCONFIGDIR"] = mpldir

    detail_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for scenario in scenarios:
        label = scenario.get("label", "scenario")
        cells = int(scenario.get("cells", 200))
        steps = int(scenario.get("steps", 300))
        concentration = float(scenario.get("concentration", 10.0))

        metrics_per_seed = []
        for seed in seeds:
            metrics = _run_scenario(
                python_exe=python_exe,
                script_path=script_path,
                env=env,
                seed=int(seed),
                cells=cells,
                steps=steps,
                concentration=concentration,
                genes=genes,
                guides=guides,
                kmax=kmax,
                num_draws=num_draws,
                batch_size=batch_size,
                lr=lr,
                clip_norm=clip_norm,
            )
            metrics_per_seed.append(metrics)
            detail_rows.append(
                {
                    "label": label,
                    "seed": int(seed),
                    "cells": cells,
                    "steps": steps,
                    "concentration": concentration,
                    **metrics,
                }
            )

        rmse_vals = [m["rmse"] for m in metrics_per_seed]
        corr_vals = [m["corr"] for m in metrics_per_seed]
        sign_vals = [m["sign"] for m in metrics_per_seed]

        rmse_mean, rmse_sd = _mean_std(rmse_vals)
        corr_mean, corr_sd = _mean_std(corr_vals)
        sign_mean, sign_sd = _mean_std(sign_vals)

        summary_rows.append(
            {
                "label": label,
                "cells": cells,
                "steps": steps,
                "concentration": concentration,
                "rmse_mean": rmse_mean,
                "rmse_sd": rmse_sd,
                "corr_mean": corr_mean,
                "corr_sd": corr_sd,
                "sign_mean": sign_mean,
                "sign_sd": sign_sd,
            }
        )

        if corr_mean < warn_corr or sign_mean < warn_sign:
            print(
                f"Warning: {label} degraded (r={corr_mean:.3f}, sign={sign_mean:.3f})."
            )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    out_summary = Path(args.out_summary)
    out_detail = Path(args.out_detail)
    _ensure_parent(out_summary)
    _ensure_parent(out_detail)
    _maybe_write(out_summary, args.force)
    _maybe_write(out_detail, args.force)

    summary_df.to_csv(out_summary, index=False)
    detail_df.to_csv(out_detail, index=False)

    print("Stress test summary")
    for row in summary_rows:
        print(
            f"{row['label']}: RMSE={row['rmse_mean']:.4f}, r={row['corr_mean']:.3f}, sign={row['sign_mean']:.3f}"
        )
    print("Wrote:", out_summary)
    print("Wrote:", out_detail)


if __name__ == "__main__":
    main()
