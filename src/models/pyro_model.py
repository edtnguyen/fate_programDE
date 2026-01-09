"""Pyro model stubs for hierarchical guide and gene effects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from pyro.infer.autoguide import AutoGuide


DEFAULT_REF_FATE = "EC"
DEFAULT_CONTRAST_FATE = "MES"


def resolve_fate_names(
    fate_names: Sequence[str],
    ref_fate: str = DEFAULT_REF_FATE,
) -> tuple[tuple[str, ...], list[str], int, list[int]]:
    """
    Validate fate names and return non-reference ordering plus indices.

    Returns
    -------
    tuple
        (fate_names, non_ref_fates, ref_idx, non_ref_indices)
    """
    fate_names = tuple(fate_names)
    if len(fate_names) != len(set(fate_names)):
        raise ValueError(f"fate_names must be unique (got {fate_names})")
    if len(fate_names) != 3:
        raise ValueError(f"Expected exactly 3 fates (got {fate_names})")
    if ref_fate not in fate_names:
        raise ValueError(f"ref_fate '{ref_fate}' not found in fate_names={fate_names}")
    non_ref_fates = [fate for fate in fate_names if fate != ref_fate]
    if len(non_ref_fates) != 2:
        raise ValueError(
            f"Expected 2 non-reference fates (ref_fate={ref_fate}, fate_names={fate_names})"
        )
    ref_idx = fate_names.index(ref_fate)
    non_ref_indices = [fate_names.index(fate) for fate in non_ref_fates]
    return fate_names, non_ref_fates, ref_idx, non_ref_indices


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

    beta_t = build_guide_effects(
        theta_t=theta_t, delta_t=delta_t, gene_of_guide_t=gene_of_guide_t
    )

    alpha_by_cell = alpha_t[:, day_t].transpose(0, 1)
    b_by_cell = b_t[:, rep_t].transpose(0, 1)
    gamma_by_cell = gamma_t.unsqueeze(0) * k_t.unsqueeze(1)

    beta_day = beta_t.permute(2, 0, 1)
    beta_day_by_cell = beta_day[day_t]
    guide_ids_exp = guide_ids_t.unsqueeze(-1).expand(-1, -1, beta_day_by_cell.shape[-1])
    beta_guides = torch.gather(beta_day_by_cell, 1, guide_ids_exp)
    guide_sum = (beta_guides * mask_t.unsqueeze(-1)).sum(dim=1)

    return alpha_by_cell + b_by_cell + gamma_by_cell + guide_sum


def construct_theta_core(
    tau: "torch.Tensor",
    z0: "torch.Tensor",
    sigma_time: "torch.Tensor | None",
    eps: "torch.Tensor | None",
    D: int,
) -> "torch.Tensor":
    """
    Construct gene-by-day effects (without the baseline gene row).
    """
    import torch

    if D <= 0:
        raise ValueError("D must be a positive integer")

    theta0 = tau[..., None, :] * z0
    if D > 1:
        if eps is None:
            raise ValueError("eps is required when D > 1")
        if sigma_time is None:
            raise ValueError("sigma_time is required when D > 1")
        increments = eps * sigma_time.unsqueeze(-3)
        theta_rest = torch.cumsum(increments, dim=-1) + theta0.unsqueeze(-1)
        theta_core = torch.cat([theta0[..., None], theta_rest], dim=-1)
    else:
        theta_core = theta0[..., None]
    return theta_core


def add_zero_gene_row(theta_core: "torch.Tensor") -> "torch.Tensor":
    """
    Prepend the baseline gene row (index 0) to theta.
    """
    import torch

    zeros = torch.zeros(
        (*theta_core.shape[:-3], 1, theta_core.shape[-2], theta_core.shape[-1]),
        device=theta_core.device,
        dtype=theta_core.dtype,
    )
    return torch.cat([zeros, theta_core], dim=-3)


def construct_delta_core(
    sigma_guide: "torch.Tensor",
    u: "torch.Tensor",
    guide_to_gene: "torch.Tensor",
    n_guides_per_gene: "torch.Tensor",
) -> "torch.Tensor":
    """
    Construct guide deviations (without the baseline guide row).
    """
    import torch

    if sigma_guide.ndim < 1:
        raise ValueError("sigma_guide must have at least 1 dimension")
    if u.ndim < 2:
        raise ValueError("u must have at least 2 dimensions")
    if guide_to_gene.ndim != 1:
        raise ValueError("guide_to_gene must be 1D: (G,)")
    if n_guides_per_gene.ndim != 1:
        raise ValueError("n_guides_per_gene must be 1D: (L,)")
    if guide_to_gene.shape[0] != u.shape[-2]:
        raise ValueError("guide_to_gene length must match G")
    if u.shape[-1] != sigma_guide.shape[-1]:
        raise ValueError("u last dim must match F*")
    if u.shape[:-2] != sigma_guide.shape[:-1]:
        raise ValueError("sigma_guide leading dims must match u leading dims")

    guide_to_gene = guide_to_gene.to(device=u.device, dtype=torch.long)
    n_guides_per_gene = n_guides_per_gene.to(device=u.device)

    fstar = sigma_guide.shape[-1]
    L = n_guides_per_gene.shape[0]

    u_t = u.transpose(-1, -2)
    guide_to_gene_exp = guide_to_gene.view(
        *([1] * (u_t.ndim - 1)), u_t.shape[-1]
    )
    guide_to_gene_exp = guide_to_gene_exp.expand(*u_t.shape)
    sum_u = torch.zeros((*u_t.shape[:-1], L), device=u.device, dtype=u.dtype)
    sum_u = sum_u.scatter_add(-1, guide_to_gene_exp, u_t)
    denom = n_guides_per_gene.to(u.dtype).clamp(min=1)
    denom = denom.view(*([1] * (sum_u.ndim - 1)), L)
    mean_u = sum_u / denom
    u_centered = u_t - mean_u.gather(-1, guide_to_gene_exp)
    sigma_exp = sigma_guide.unsqueeze(-1)
    delta_core = (sigma_exp * u_centered).transpose(-1, -2)
    return delta_core


def add_zero_guide_row(delta_core: "torch.Tensor") -> "torch.Tensor":
    """
    Prepend the baseline guide row (index 0) to delta.
    """
    import torch

    zeros = torch.zeros(
        (*delta_core.shape[:-2], 1, delta_core.shape[-1]),
        device=delta_core.device,
        dtype=delta_core.dtype,
    )
    return torch.cat([zeros, delta_core], dim=-2)


def fate_model(
    p_t: "torch.Tensor",
    day_t: "torch.Tensor",
    rep_t: "torch.Tensor",
    k_t: "torch.Tensor",
    guide_ids_t: "torch.Tensor",
    mask_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
    guide_to_gene_t: "torch.Tensor",
    n_guides_per_gene_t: "torch.Tensor",
    *,
    fate_names: Sequence[str],
    ref_fate: str = DEFAULT_REF_FATE,
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
    likelihood_weight: float = 1.0,
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

    fate_names, non_ref_fates, ref_idx, _ = resolve_fate_names(
        fate_names, ref_fate=ref_fate
    )
    if p_t.ndim != 2 or p_t.shape[1] != len(fate_names):
        raise ValueError(f"p_t must be (N,{len(fate_names)}) ordered as {fate_names}")
    if day_t.ndim != 1 or rep_t.ndim != 1 or k_t.ndim != 1:
        raise ValueError("day_t, rep_t, and k_t must be 1D tensors")
    if guide_ids_t.shape != mask_t.shape:
        raise ValueError("guide_ids_t and mask_t must have the same shape")
    if guide_ids_t.shape[1] != Kmax:
        raise ValueError(
            f"guide_ids_t second dim {guide_ids_t.shape[1]} does not match Kmax={Kmax}"
        )

    device = p_t.device
    N = p_t.shape[0]
    fstar = len(non_ref_fates)

    alpha = pyro.sample(
        "alpha", dist.Normal(0.0, s_alpha).expand([fstar, D]).to_event(2)
    )

    b = pyro.sample("b", dist.Normal(0.0, s_rep).expand([fstar, R]).to_event(2))

    gamma = pyro.sample("gamma", dist.Normal(0.0, s_gamma).expand([fstar]).to_event(1))

    tau = pyro.sample("tau", dist.HalfNormal(s_tau).expand([fstar]).to_event(1))
    z0 = pyro.sample("z0", dist.Normal(0.0, 1.0).expand([L, fstar]).to_event(2))

    sigma_time = None
    eps = None
    if D > 1:
        sigma_time = pyro.sample(
            "sigma_time", dist.HalfNormal(s_time).expand([fstar, D - 1]).to_event(2)
        )
        eps = pyro.sample(
            "eps", dist.Normal(0.0, 1.0).expand([L, fstar, D - 1]).to_event(3)
        )

    theta_core = construct_theta_core(
        tau=tau,
        z0=z0,
        sigma_time=sigma_time,
        eps=eps,
        D=D,
    )
    theta = add_zero_gene_row(theta_core)

    sigma_guide = pyro.sample(
        "sigma_guide", dist.HalfNormal(s_guide).expand([fstar]).to_event(1)
    )
    u = pyro.sample("u", dist.Normal(0.0, 1.0).expand([G, fstar]).to_event(2))
    delta_core = construct_delta_core(
        sigma_guide=sigma_guide,
        u=u,
        guide_to_gene=guide_to_gene_t,
        n_guides_per_gene=n_guides_per_gene_t,
    )
    delta = add_zero_guide_row(delta_core)

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

        zeros = torch.zeros((p_batch.shape[0], 1), device=device)
        eta_parts = []
        nonref_col = 0
        for idx in range(len(fate_names)):
            if idx == ref_idx:
                eta_parts.append(zeros)
            else:
                eta_parts.append(eta_nonref[:, nonref_col : nonref_col + 1])
                nonref_col += 1
        eta = torch.cat(eta_parts, dim=1)
        log_pi = F.log_softmax(eta, dim=-1)

        logp = (p_batch * log_pi).sum(-1)
        scale = float(N) / float(logp.shape[0])
        pyro.factor("soft_label_loglik", logp.sum() * scale * likelihood_weight)


def fit_svi(
    p_t: "torch.Tensor",
    day_t: "torch.Tensor",
    rep_t: "torch.Tensor",
    k_t: "torch.Tensor",
    guide_ids_t: "torch.Tensor",
    mask_t: "torch.Tensor",
    gene_of_guide_t: "torch.Tensor",
    guide_to_gene_t: "torch.Tensor",
    n_guides_per_gene_t: "torch.Tensor",
    *,
    fate_names: Sequence[str],
    ref_fate: str = DEFAULT_REF_FATE,
    L: int,
    G: int,
    D: int,
    R: int,
    Kmax: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    num_steps: int,
    s_alpha: float = 1.0,
    s_rep: float = 1.0,
    s_gamma: float = 1.0,
    s_tau: float = 1.0,
    s_time: float = 1.0,
    s_guide: float = 1.0,
    likelihood_weight: float = 1.0,
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
        return fate_model(
            *args,
            **kwargs,
            subsample_size=subsample_size,
            fate_names=fate_names,
            ref_fate=ref_fate,
            s_alpha=s_alpha,
            s_rep=s_rep,
            s_gamma=s_gamma,
            s_tau=s_tau,
            s_time=s_time,
            s_guide=s_guide,
            likelihood_weight=likelihood_weight,
        )

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
            guide_to_gene_t,
            n_guides_per_gene_t,
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

    (
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    ) = model_args
    G = int(gene_of_guide_t.shape[0] - 1)
    R = int(rep_t.max().item() + 1)
    Kmax = int(guide_ids_t.shape[1])

    return_sites = ("tau", "z0")
    if D > 1:
        return_sites = return_sites + ("sigma_time", "eps")
    predictive = Predictive(guide, num_samples=num_draws, return_sites=return_sites)
    samples = predictive(
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
        L=L,
        G=G,
        D=D,
        R=R,
        Kmax=Kmax,
    )

    theta_core = construct_theta_core(
        tau=samples["tau"],
        z0=samples["z0"],
        sigma_time=samples.get("sigma_time"),
        eps=samples.get("eps"),
        D=D,
    )
    theta = add_zero_gene_row(theta_core)

    return theta.detach().cpu().numpy()


def reconstruct_delta_samples(
    guide: "AutoGuide",
    model_args: tuple["torch.Tensor", ...],
    *,
    L: int,
    D: int,
    num_draws: int,
) -> np.ndarray:
    """
    Draw posterior samples of guide deviations from the fitted guide.

    Returns
    -------
    np.ndarray
        Array of sampled delta values with shape (S, G+1, F*).
    """
    import torch
    from pyro.infer import Predictive

    if D <= 0 or L <= 0:
        raise ValueError("L and D must be positive integers")
    if num_draws <= 0:
        raise ValueError("num_draws must be a positive integer")

    (
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
    ) = model_args
    G = int(gene_of_guide_t.shape[0] - 1)
    R = int(rep_t.max().item() + 1)
    Kmax = int(guide_ids_t.shape[1])

    return_sites = ("sigma_guide", "u")
    predictive = Predictive(guide, num_samples=num_draws, return_sites=return_sites)
    samples = predictive(
        p_t,
        day_t,
        rep_t,
        k_t,
        guide_ids_t,
        mask_t,
        gene_of_guide_t,
        guide_to_gene_t,
        n_guides_per_gene_t,
        L=L,
        G=G,
        D=D,
        R=R,
        Kmax=Kmax,
    )

    delta_core = construct_delta_core(
        sigma_guide=samples["sigma_guide"],
        u=samples["u"],
        guide_to_gene=guide_to_gene_t,
        n_guides_per_gene=n_guides_per_gene_t,
    )
    delta = add_zero_guide_row(delta_core)

    return delta.detach().cpu().numpy()


def export_gene_summary_for_ash(
    *,
    guide: "AutoGuide",
    model_args: tuple["torch.Tensor", ...],
    gene_names: Sequence[str],
    fate_names: Sequence[str],
    ref_fate: str = DEFAULT_REF_FATE,
    contrast_fate: str = DEFAULT_CONTRAST_FATE,
    L: int,
    D: int,
    num_draws: int,
    day_cell_counts: Sequence[int],
    weights: Optional[Sequence[float]],
    out_csv: str | None,
) -> pd.DataFrame:
    """
    Summarize gene effects across days for the contrast fate and write an ash-ready CSV.

    Returns
    -------
    pd.DataFrame
        Gene-level summary table, optionally written to ``out_csv`` when provided.
    """
    if len(gene_names) != L:
        raise ValueError(f"gene_names length {len(gene_names)} does not match L={L}")
    if D <= 0 or L <= 0:
        raise ValueError("L and D must be positive integers")

    _, non_ref_fates, _, _ = resolve_fate_names(fate_names, ref_fate=ref_fate)
    if contrast_fate == ref_fate:
        raise ValueError("contrast_fate must differ from ref_fate")
    if contrast_fate not in non_ref_fates:
        raise ValueError(
            f"contrast_fate '{contrast_fate}' not in non-reference fates {non_ref_fates}"
        )
    contrast_idx = non_ref_fates.index(contrast_fate)

    theta_samples = reconstruct_theta_samples(
        guide=guide,
        model_args=model_args,
        L=L,
        D=D,
        num_draws=num_draws,
    )

    contrast_samples = theta_samples[:, 1:, contrast_idx, :]

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

    weighted = np.tensordot(contrast_samples, weights_arr, axes=([2], [0]))
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
    if out_csv is not None:
        summary_df.to_csv(out_csv, index=False)
    return summary_df


def export_gene_summary_for_mash(
    *,
    guide: "AutoGuide",
    model_args: tuple["torch.Tensor", ...],
    gene_names: Sequence[str],
    fate_names: Sequence[str],
    ref_fate: str = DEFAULT_REF_FATE,
    contrast_fate: str = DEFAULT_CONTRAST_FATE,
    L: int,
    D: int,
    num_draws: int,
    out_csv: str | None,
) -> pd.DataFrame:
    """
    Export daywise MES–EC gene effects for mashr.

    Returns
    -------
    pd.DataFrame
        Gene-level summary table with daywise betahat/se columns.
    """
    if len(gene_names) != L:
        raise ValueError(f"gene_names length {len(gene_names)} does not match L={L}")
    if D <= 0 or L <= 0:
        raise ValueError("L and D must be positive integers")

    _, non_ref_fates, _, _ = resolve_fate_names(fate_names, ref_fate=ref_fate)
    if contrast_fate == ref_fate:
        raise ValueError("contrast_fate must differ from ref_fate")
    if contrast_fate not in non_ref_fates:
        raise ValueError(
            f"contrast_fate '{contrast_fate}' not in non-reference fates {non_ref_fates}"
        )
    contrast_idx = non_ref_fates.index(contrast_fate)

    theta_samples = reconstruct_theta_samples(
        guide=guide,
        model_args=model_args,
        L=L,
        D=D,
        num_draws=num_draws,
    )

    contrast_samples = theta_samples[:, 1:, contrast_idx, :]
    betahat = contrast_samples.mean(axis=0)
    sebetahat = contrast_samples.std(axis=0, ddof=0)

    summary_df = pd.DataFrame({"gene": list(gene_names)})
    for d in range(D):
        summary_df[f"betahat_d{d}"] = betahat[:, d]
    for d in range(D):
        summary_df[f"se_d{d}"] = sebetahat[:, d]
    if out_csv is not None:
        summary_df.to_csv(out_csv, index=False)
    return summary_df
