"""Pyro model stubs for hierarchical guide and gene effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from pyro.infer.autoguide import AutoGuide


FATE_NAMES = ("EC", "MES", "NEU")
REF_FATE = "EC"
NON_REF_FATES = ("MES", "NEU")


@dataclass(frozen=True)
class LatentIndex:
    """Indexing metadata for latent parameters in the fate model."""

    L: int
    G: int
    D: int
    R: int

    def __post_init__(self) -> None:
        for name, value in (("L", self.L), ("G", self.G), ("D", self.D), ("R", self.R)):
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer (got {value})")

    def fate_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(FATE_NAMES)}

    def non_ref_idx(self) -> tuple[int, int]:
        fate_to_idx = self.fate_to_idx()
        return (fate_to_idx[NON_REF_FATES[0]], fate_to_idx[NON_REF_FATES[1]])

    def shapes(self) -> dict[str, tuple[int, ...]]:
        fstar = len(NON_REF_FATES)
        return {
            "theta": (self.L + 1, fstar, self.D),
            "delta": (self.G + 1, fstar),
            "alpha": (fstar, self.D),
            "b": (fstar, self.R),
            "gamma": (fstar,),
        }


def build_guide_effects(
    theta_t: "torch.Tensor",
    delta_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
) -> "torch.Tensor":
    """
    Build guide-by-fate-by-day effects via beta_g = theta_gene(g) + delta_g.

    Returns
    -------
    torch.Tensor
        beta tensor of shape (G+1, F*, D) with baseline g=0 set to zero.
    """
    import torch

    if theta_t.ndim != 3:
        raise ValueError("theta_t must be 3D: (L+1, F*, D)")
    if delta_t.ndim != 2:
        raise ValueError("delta_t must be 2D: (G+1, F*)")
    if gene_of_guide_t.ndim != 1:
        raise ValueError("gene_of_guide_t must be 1D: (G+1,)")

    theta_by_guide = theta_t[gene_of_guide_t]
    beta_t = theta_by_guide + delta_t.unsqueeze(-1)

    if beta_t.shape[0] > 0:
        beta_t = beta_t.clone()
        beta_t[0] = torch.zeros_like(beta_t[0])

    return beta_t


def compute_linear_predictor(
    alpha_t: "torch.Tensor",
    b_t: "torch.Tensor",
    gamma_t: "torch.Tensor",
    k_t: "torch.Tensor",
    guide_ids_t: "torch.Tensor",
    mask_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
    theta_t: "torch.Tensor",
    delta_t: "torch.Tensor",
    day_t: "torch.Tensor",
    rep_t: "torch.Tensor",
) -> "torch.Tensor":
    """
    Compute eta_i,f* using day, replicate, burden, and guide sums.

    Returns
    -------
    torch.Tensor
        Linear predictor of shape (N, F*) for non-reference fates.
    """
    import torch

    if guide_ids_t.shape != mask_t.shape:
        raise ValueError("guide_ids_t and mask_t must have the same shape")
    if day_t.ndim != 1 or rep_t.ndim != 1:
        raise ValueError("day_t and rep_t must be 1D tensors")

    beta_t = build_guide_effects(theta_t=theta_t, delta_t=delta_t, gene_of_guide_t=gene_of_guide_t)

    alpha_by_cell = alpha_t[:, day_t].transpose(0, 1)
    b_by_cell = b_t[:, rep_t].transpose(0, 1)
    gamma_by_cell = gamma_t.unsqueeze(0) * k_t.unsqueeze(1)

    beta_day = beta_t.permute(2, 0, 1)
    beta_day_by_cell = beta_day[day_t]
    guide_ids_exp = guide_ids_t.unsqueeze(-1).expand(-1, -1, beta_day_by_cell.shape[-1])
    beta_guides = torch.gather(beta_day_by_cell, 1, guide_ids_exp)
    guide_sum = (beta_guides * mask_t.unsqueeze(-1)).sum(dim=1)

    return alpha_by_cell + b_by_cell + gamma_by_cell + guide_sum


def primary_contrast_mes_ec(theta_t: "torch.Tensor") -> "torch.Tensor":
    """
    Extract the MES–EC primary contrast from theta.

    Returns
    -------
    torch.Tensor
        MES effects with shape (L+1, D), including baseline row 0.
    """
    if theta_t.ndim != 3:
        raise ValueError("theta_t must be 3D: (L+1, F*, D)")
    mes_idx = 0
    return theta_t[:, mes_idx, :]


def fate_model(
    p_t: "torch.Tensor",
    day_t: "torch.Tensor",
    rep_t: "torch.Tensor",
    k_t: "torch.Tensor",
    guide_ids_t: "torch.Tensor",
    mask_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
    *,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    s_alpha: float = 1.0,
    s_rep: float = 1.0,
    s_gamma: float = 1.0,
    s_tau: float = 1.0,
    s_time: float = 1.0,
    s_guide: float = 1.0,
    subsample_size: int | None = None,
) -> None:
    """
    Define the hierarchical Pyro model for MOI-aware fate probabilities.

    Returns
    -------
    None
        The Pyro model does not return values; it defines a probabilistic
        program for inference.
    """
    import pyro
    import pyro.distributions as dist
    import torch
    import torch.nn.functional as F

    if p_t.ndim != 2 or p_t.shape[1] != len(FATE_NAMES):
        raise ValueError(
            f"p_t must be (N,{len(FATE_NAMES)}) ordered as {FATE_NAMES}"
        )
    if day_t.ndim != 1 or rep_t.ndim != 1 or k_t.ndim != 1:
        raise ValueError("day_t, rep_t, and k_t must be 1D tensors")
    if guide_ids_t.shape != mask_t.shape:
        raise ValueError("guide_ids_t and mask_t must have the same shape")

    device = p_t.device
    N = p_t.shape[0]
    fstar = len(NON_REF_FATES)

    sigma_alpha = pyro.sample(
        "sigma_alpha", dist.HalfNormal(s_alpha).expand([fstar]).to_event(1)
    )
    alpha = pyro.sample(
        "alpha",
        dist.Normal(torch.zeros((fstar, D), device=device), sigma_alpha.unsqueeze(-1)).to_event(2),
    )

    sigma_rep = pyro.sample(
        "sigma_rep", dist.HalfNormal(s_rep).expand([fstar]).to_event(1)
    )
    b = pyro.sample(
        "b",
        dist.Normal(torch.zeros((fstar, R), device=device), sigma_rep.unsqueeze(-1)).to_event(2),
    )

    sigma_gamma = pyro.sample(
        "sigma_gamma", dist.HalfNormal(s_gamma).expand([fstar]).to_event(1)
    )
    gamma = pyro.sample(
        "gamma",
        dist.Normal(torch.zeros((fstar,), device=device), sigma_gamma).to_event(1),
    )

    tau = pyro.sample("tau", dist.HalfNormal(s_tau).expand([fstar]).to_event(1))
    z0 = pyro.sample(
        "z0", dist.Normal(0.0, 1.0).expand([L, fstar]).to_event(2)
    )
    theta0 = tau.view(1, fstar) * z0

    sigma_time = pyro.sample(
        "sigma_time", dist.HalfNormal(s_time).expand([fstar]).to_event(1)
    )
    if D > 1:
        eps = pyro.sample(
            "eps", dist.Normal(0.0, 1.0).expand([L, fstar, D - 1]).to_event(3)
        )
        increments = sigma_time.view(1, fstar, 1) * eps
        theta_rest = torch.cumsum(increments, dim=2) + theta0.unsqueeze(-1)
        theta_core = torch.cat([theta0.unsqueeze(-1), theta_rest], dim=2)
    else:
        theta_core = theta0.unsqueeze(-1)

    theta = torch.cat(
        [torch.zeros((1, fstar, D), device=device), theta_core], dim=0
    )

    sigma_guide = pyro.sample(
        "sigma_guide", dist.HalfNormal(s_guide).expand([fstar]).to_event(1)
    )
    u = pyro.sample(
        "u", dist.Normal(0.0, 1.0).expand([G, fstar]).to_event(2)
    )
    delta_core = sigma_guide.view(1, fstar) * u
    delta = torch.cat(
        [torch.zeros((1, fstar), device=device), delta_core], dim=0
    )

    batch_size = subsample_size if subsample_size is not None else N
    batch_size = min(batch_size, N)

    with pyro.plate("cells", N, subsample_size=batch_size) as idx:
        p_batch = p_t[idx]
        day_batch = day_t[idx]
        rep_batch = rep_t[idx]
        k_batch = k_t[idx]
        guide_ids_batch = guide_ids_t[idx]
        mask_batch = mask_t[idx]

        eta_nonref = compute_linear_predictor(
            alpha_t=alpha,
            b_t=b,
            gamma_t=gamma,
            k_t=k_batch,
            guide_ids_t=guide_ids_batch,
            mask_t=mask_batch,
            gene_of_guide_t=gene_of_guide_t,
            theta_t=theta,
            delta_t=delta,
            day_t=day_batch,
            rep_t=rep_batch,
        )

        eta = torch.zeros((p_batch.shape[0], len(FATE_NAMES)), device=device)
        eta[:, 1:] = eta_nonref
        log_pi = F.log_softmax(eta, dim=-1)

        logp = (p_batch * log_pi).sum(-1)
        pyro.factor("soft_label_loglik", logp)


def fit_svi(
    p_t: "torch.Tensor",
    day_t: "torch.Tensor",
    rep_t: "torch.Tensor",
    k_t: "torch.Tensor",
    guide_ids_t: "torch.Tensor",
    mask_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
    *,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    num_steps: int,
    seed: int = 0,
) -> "AutoGuide":
    """
    Fit the model with SVI and return the variational guide.

    Returns
    -------
    AutoGuide
        A Pyro guide object encapsulating the variational approximation.
    """
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import ClippedAdam

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    N = p_t.shape[0]
    subsample_size = min(batch_size, N) if batch_size else N

    def model(*args, **kwargs):
        return fate_model(*args, **kwargs, subsample_size=subsample_size)

    guide = AutoNormal(model)
    optim = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    for _ in range(num_steps):
        svi.step(
            p_t,
            day_t,
            rep_t,
            k_t,
            guide_ids_t,
            mask_t,
            gene_of_guide_t,
            L=L,
            G=G,
            D=D,
            R=R,
            Kmax=Kmax,
        )

    return guide


def reconstruct_theta_samples(
    guide: "AutoGuide",
    model_args: tuple["torch.Tensor", ...],
    *,
    L: int,
    D: int,
    num_draws: int,
) -> np.ndarray:
    """
    Draw posterior samples of gene-by-day effects from the fitted guide.

    Returns
    -------
    np.ndarray
        Array of sampled theta values with shape (S, L+1, F*, D).
    """
    import torch
    from pyro.infer import Predictive

    if D <= 0 or L <= 0:
        raise ValueError("L and D must be positive integers")
    if num_draws <= 0:
        raise ValueError("num_draws must be a positive integer")

    p_t, day_t, rep_t, k_t, guide_ids_t, mask_t, gene_of_guide_t = model_args
    G = int(gene_of_guide_t.shape[0] - 1)
    R = int(rep_t.max().item() + 1)
    Kmax = int(guide_ids_t.shape[1])

    return_sites = ("tau", "z0", "sigma_time", "eps")
    predictive = Predictive(guide, num_samples=num_draws, return_sites=return_sites)
    samples = predictive(
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        L=L,
        G=G,
        D=D,
        R=R,
        Kmax=Kmax,
    )

    tau = samples["tau"]
    z0 = samples["z0"]
    sigma_time = samples["sigma_time"]
    fstar = tau.shape[-1]

    theta0 = tau[:, None, :] * z0
    if D > 1:
        eps = samples["eps"]
        increments = sigma_time[:, None, :, None] * eps
        theta_rest = torch.cumsum(increments, dim=-1) + theta0[..., None]
        theta_core = torch.cat([theta0[..., None], theta_rest], dim=-1)
    else:
        theta_core = theta0[..., None]

    zeros = torch.zeros((num_draws, 1, fstar, D), device=theta_core.device)
    theta = torch.cat([zeros, theta_core], dim=1)

    return theta.detach().cpu().numpy()


def export_gene_summary_for_ash(
    *,
    guide: "AutoGuide",
    model_args: tuple["torch.Tensor", ...],
    gene_names: Sequence[str],
    L: int,
    D: int,
    num_draws: int,
    day_cell_counts: Sequence[int],
    weights: Optional[Sequence[float]],
    out_csv: str,
) -> pd.DataFrame:
    """
    Summarize gene effects across days and write an ash-ready CSV.

    Returns
    -------
    pd.DataFrame
        Gene-level summary table written to ``out_csv``.
    """
    if len(gene_names) != L:
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match L={L}"
        )
    if D <= 0 or L <= 0:
        raise ValueError("L and D must be positive integers")

    theta_samples = reconstruct_theta_samples(
        guide=guide,
        model_args=model_args,
        L=L,
        D=D,
        num_draws=num_draws,
    )

    mes_samples = theta_samples[:, 1:, 0, :]

    if weights is None:
        weights_arr = np.asarray(day_cell_counts, dtype=np.float64)
        if weights_arr.shape[0] != D:
            raise ValueError("day_cell_counts length must match D")
        weight_sum = weights_arr.sum()
        if weight_sum <= 0:
            raise ValueError("day_cell_counts must sum to a positive value")
        weights_arr = weights_arr / weight_sum
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape[0] != D:
            raise ValueError("weights length must match D")
        weight_sum = weights_arr.sum()
        if weight_sum <= 0:
            raise ValueError("weights must sum to a positive value")
        weights_arr = weights_arr / weight_sum

    weighted = np.tensordot(mes_samples, weights_arr, axes=([2], [0]))
    betahat = weighted.mean(axis=0)
    sebetahat = weighted.std(axis=0, ddof=1)

    summary_df = pd.DataFrame(
        {
            "gene": list(gene_names),
            "betahat": betahat,
            "sebetahat": sebetahat,
        }
    )
    for d in range(D):
        summary_df[f"w{d}"] = float(weights_arr[d])
    summary_df.to_csv(out_csv, index=False)
    return summary_df
