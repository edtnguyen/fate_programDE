import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts import pyro_io  # noqa: E402

try:
    import anndata as ad
except Exception:  # pragma: no cover - handled via skip
    ad = None


class TestPyroIOHelpers(unittest.TestCase):
    def test_parse_day_to_int(self):
        days = pd.Series(["d0", "d1", "d2", "d3"])
        out = pyro_io.parse_day_to_int(days, D=4)
        np.testing.assert_array_equal(out, np.array([0, 1, 2, 3]))

    def test_parse_rep_to_int(self):
        reps = np.array(["A", "B", "A"])
        out = pyro_io.parse_rep_to_int(reps, R=2)
        np.testing.assert_array_equal(out, np.array([0, 1, 0]))

    def test_get_fate_probs_dataframe(self):
        df = pd.DataFrame(
            {
                "EC": [0.2, 0.1],
                "MES": [0.3, 0.4],
                "NEU": [0.5, 0.5],
            }
        )
        p = pyro_io.get_fate_probs(type("A", (), {"obsm": {"f": df}}), "f", ["EC", "MES", "NEU"])
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_allclose(p.sum(axis=1), np.ones(2))

    def test_build_id_maps(self):
        guide_names = ["g1", "g2", "ntc1"]
        guide_map_df = pd.DataFrame(
            {
                "guide_name": ["g1", "g2", "ntc1"],
                "gene_name": ["GeneA", "GeneB", "NTC"],
                "is_ntc": [0, 0, 1],
            }
        )
        guide_name_to_gid, gid_to_gene, gene_names, L, G = pyro_io.build_id_maps(
            guide_names, guide_map_df
        )
        self.assertEqual(guide_name_to_gid["g1"], 1)
        self.assertEqual(guide_name_to_gid["g2"], 2)
        self.assertEqual(guide_name_to_gid["ntc1"], 0)
        self.assertEqual(gene_names, ["GeneA", "GeneB"])
        self.assertEqual(L, 2)
        self.assertEqual(G, 2)
        np.testing.assert_array_equal(gid_to_gene, np.array([0, 1, 2]))

    def test_guides_to_padded_from_csr_raises(self):
        guide_names = ["g1", "g2", "ntc1"]
        guide_name_to_gid = {"g1": 1, "g2": 2, "ntc1": 0}
        data = np.array([1, 1, 1, 1, 1])
        rows = np.array([0, 0, 1, 1, 1])
        cols = np.array([0, 1, 0, 1, 2])
        mat = sp.csr_matrix((data, (rows, cols)), shape=(2, 3))
        with self.assertRaises(ValueError):
            pyro_io.guides_to_padded_from_csr(mat, guide_names, guide_name_to_gid, Kmax=2)


@unittest.skipIf(ad is None, "anndata is not available")
class TestLoadAdataInputs(unittest.TestCase):
    def test_load_adata_inputs_filters_k(self):
        obs = pd.DataFrame({"day": ["d0", "d0"], "cell": ["c1", "c2"]})
        covar = pd.DataFrame({"rep": ["r1", "r1"]})
        fate = pd.DataFrame(
            {
                "EC": [0.2, 0.2],
                "MES": [0.3, 0.3],
                "NEU": [0.5, 0.5],
            }
        )
        guide = pd.DataFrame(
            {
                "g1": [1, 1],
                "g2": [1, 1],
                "g3": [0, 1],
            }
        )
        adata = ad.AnnData(X=np.zeros((2, 1)))
        adata.obs = obs
        adata.obsm["covar"] = covar
        adata.obsm["fate"] = fate
        adata.obsm["guide"] = guide

        guide_map = pd.DataFrame(
            {
                "guide_name": ["g1", "g2", "g3"],
                "gene_name": ["A", "B", "C"],
                "is_ntc": [0, 0, 0],
            }
        )

        cfg = {
            "day_key": "day",
            "covar_key": "covar",
            "rep_key": "rep",
            "fate_prob_key": "fate",
            "fates": ["EC", "MES", "NEU"],
            "guide_key": "guide",
            "Kmax": 2,
            "D": 1,
            "R": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            guide_map_path = Path(tmpdir) / "guide_map.csv"
            guide_map.to_csv(guide_map_path, index=False)
            cell_df, p, guide_ids, mask, _, _, _, _, _ = pyro_io.load_adata_inputs(
                adata, cfg, guide_map_csv=str(guide_map_path)
            )
        self.assertEqual(len(cell_df), 1)
        self.assertEqual(p.shape[0], 1)
        self.assertEqual(guide_ids.shape[0], 1)
        self.assertEqual(mask.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
