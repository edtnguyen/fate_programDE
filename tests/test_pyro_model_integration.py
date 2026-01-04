import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    import torch
    import pyro
except Exception:  # pragma: no cover - handled via skip
    torch = None
    pyro = None

pyro_model_path = ROOT / "src" / "models" / "pyro_model.py"
spec = importlib.util.spec_from_file_location("pyro_model", pyro_model_path)
pyro_model = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"Unable to load {pyro_model_path}")
spec.loader.exec_module(pyro_model)


@unittest.skipIf(torch is None or pyro is None, "pyro/torch is not available")
class TestPyroModelFlow(unittest.TestCase):
    def test_fit_and_export(self):
        torch.manual_seed(0)
        pyro.set_rng_seed(0)

        N = 8
        D = 2
        R = 1
        L = 2
        G = 3
        Kmax = 3

        p = np.random.default_rng(0).uniform(0.1, 1.0, size=(N, 3)).astype(np.float32)
        p = p / p.sum(axis=1, keepdims=True)
        p_t = torch.tensor(p, dtype=torch.float32)

        day_t = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        rep_t = torch.zeros(N, dtype=torch.long)

        guide_ids = np.array(
            [
                [1, 2, 0],
                [2, 3, 0],
                [1, 0, 0],
                [3, 0, 0],
                [1, 2, 3],
                [2, 0, 0],
                [3, 0, 0],
                [1, 2, 0],
            ],
            dtype=np.int64,
        )
        mask = (guide_ids > 0).astype(np.float32)
        gids_t = torch.tensor(guide_ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float32)

        gene_of_guide = np.array([0, 1, 1, 2], dtype=np.int64)
        gene_of_guide_t = torch.tensor(gene_of_guide, dtype=torch.long)
        guide_to_gene = gene_of_guide[1:] - 1
        n_guides_per_gene = np.bincount(guide_to_gene, minlength=L).astype(np.int64)
        guide_to_gene_t = torch.tensor(guide_to_gene, dtype=torch.long)
        n_guides_per_gene_t = torch.tensor(n_guides_per_gene, dtype=torch.long)

        k_t = torch.tensor(mask.sum(axis=1), dtype=torch.float32)

        guide = pyro_model.fit_svi(
            p_t,
            day_t,
            rep_t,
            k_t,
            gids_t,
            mask_t,
            gene_of_guide_t,
            guide_to_gene_t,
            n_guides_per_gene_t,
            fate_names=("EC", "MES", "NEU"),
            ref_fate="EC",
            L=L,
            G=G,
            D=D,
            R=R,
            Kmax=Kmax,
            batch_size=4,
            lr=1e-2,
            clip_norm=5.0,
            num_steps=5,
            seed=0,
        )

        summary = pyro_model.export_gene_summary_for_ash(
            guide=guide,
            model_args=(
                p_t,
                day_t,
                rep_t,
                k_t,
                gids_t,
                mask_t,
                gene_of_guide_t,
                guide_to_gene_t,
                n_guides_per_gene_t,
            ),
            gene_names=["GeneA", "GeneB"],
            fate_names=("EC", "MES", "NEU"),
            ref_fate="EC",
            contrast_fate="MES",
            L=L,
            D=D,
            num_draws=10,
            day_cell_counts=[4, 4],
            weights=None,
            out_csv=None,
        )

        self.assertEqual(len(summary), L)
        self.assertTrue({"gene", "betahat", "sebetahat"}.issubset(summary.columns))
        self.assertTrue(np.isfinite(summary["betahat"]).all())
        self.assertTrue(np.isfinite(summary["sebetahat"]).all())


if __name__ == "__main__":
    unittest.main()
