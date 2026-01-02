import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

pyro_model_path = ROOT / "src" / "models" / "pyro_model.py"
spec = importlib.util.spec_from_file_location("pyro_model", pyro_model_path)
pyro_model = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"Unable to load {pyro_model_path}")
spec.loader.exec_module(pyro_model)

resolve_fate_names = pyro_model.resolve_fate_names
add_zero_gene_row = pyro_model.add_zero_gene_row
add_zero_guide_row = pyro_model.add_zero_guide_row
compute_linear_predictor = pyro_model.compute_linear_predictor
construct_delta_core = pyro_model.construct_delta_core
construct_theta_core = pyro_model.construct_theta_core

try:
    import torch
except Exception:  # pragma: no cover - handled via skip
    torch = None


class TestResolveFateNames(unittest.TestCase):
    def test_valid(self):
        fates = ("EC", "MES", "NEU")
        fate_names, non_ref, ref_idx, non_ref_idx = resolve_fate_names(
            fates, ref_fate="EC"
        )
        self.assertEqual(fate_names, fates)
        self.assertEqual(non_ref, ["MES", "NEU"])
        self.assertEqual(ref_idx, 0)
        self.assertEqual(non_ref_idx, [1, 2])

    def test_invalid_duplicate(self):
        with self.assertRaises(ValueError):
            resolve_fate_names(("EC", "EC", "NEU"), ref_fate="EC")


@unittest.skipIf(torch is None, "torch is not available")
class TestThetaDeltaConstruction(unittest.TestCase):
    def test_construct_theta_core_d2(self):
        tau = torch.tensor([2.0, 1.0])
        z0 = torch.tensor([[1.0, 3.0]])
        sigma_time = torch.tensor([0.5, 2.0])
        eps = torch.ones((1, 2, 1))

        theta_core = construct_theta_core(
            tau=tau, z0=z0, sigma_time=sigma_time, eps=eps, D=2
        )
        self.assertEqual(theta_core.shape, (1, 2, 2))
        expected = torch.tensor([[[2.0, 2.5], [3.0, 5.0]]])
        self.assertTrue(torch.allclose(theta_core, expected))

        theta = add_zero_gene_row(theta_core)
        self.assertEqual(theta.shape, (2, 2, 2))
        self.assertTrue(torch.allclose(theta[0], torch.zeros_like(theta[0])))

    def test_construct_theta_core_d1(self):
        tau = torch.tensor([1.0, 2.0])
        z0 = torch.tensor([[3.0, 4.0]])
        sigma_time = torch.tensor([0.3, 0.7])

        theta_core = construct_theta_core(
            tau=tau, z0=z0, sigma_time=sigma_time, eps=None, D=1
        )
        self.assertEqual(theta_core.shape, (1, 2, 1))
        expected = torch.tensor([[[3.0], [8.0]]])
        self.assertTrue(torch.allclose(theta_core, expected))

    def test_construct_theta_core_time_scale(self):
        tau = torch.tensor([1.0])
        z0 = torch.tensor([[1.0]])
        sigma_time = torch.tensor([1.0])
        eps = torch.ones((1, 1, 2))

        theta_core = construct_theta_core(
            tau=tau,
            z0=z0,
            sigma_time=sigma_time,
            eps=eps,
            time_scale=[1.0, 2.0],
            D=3,
        )
        expected = torch.tensor([[[1.0, 2.0, 4.0]]])
        self.assertTrue(torch.allclose(theta_core, expected))

    def test_construct_delta_core(self):
        sigma_guide = torch.tensor([0.5, 2.0])
        u = torch.tensor([[2.0, -1.0]])
        delta_core = construct_delta_core(sigma_guide=sigma_guide, u=u)
        expected = torch.tensor([[1.0, -2.0]])
        self.assertTrue(torch.allclose(delta_core, expected))

        delta = add_zero_guide_row(delta_core)
        self.assertEqual(delta.shape, (2, 2))
        self.assertTrue(torch.allclose(delta[0], torch.zeros_like(delta[0])))

    def test_compute_linear_predictor(self):
        alpha_t = torch.tensor([[0.1], [0.2]])
        b_t = torch.tensor([[0.3], [0.4]])
        gamma_t = torch.tensor([0.5, 0.6])
        k_t = torch.tensor([2.0])
        guide_ids_t = torch.tensor([[1, 0]])
        mask_t = torch.tensor([[1.0, 0.0]])
        gene_of_guide_t = torch.tensor([0, 1])

        theta_t = torch.zeros((2, 2, 1))
        theta_t[1, :, 0] = torch.tensor([1.0, 2.0])
        delta_t = torch.zeros((2, 2))
        delta_t[1] = torch.tensor([0.1, 0.2])

        day_t = torch.tensor([0])
        rep_t = torch.tensor([0])

        eta = compute_linear_predictor(
            alpha_t=alpha_t,
            b_t=b_t,
            gamma_t=gamma_t,
            k_t=k_t,
            guide_ids_t=guide_ids_t,
            mask_t=mask_t,
            gene_of_guide_t=gene_of_guide_t,
            theta_t=theta_t,
            delta_t=delta_t,
            day_t=day_t,
            rep_t=rep_t,
        )
        expected = torch.tensor([[2.5, 4.0]])
        self.assertTrue(torch.allclose(eta, expected))


if __name__ == "__main__":
    unittest.main()
