"""Nonconvex baseline for comparison with waveX.

An alternating circulant initialization followed by smoothed amplitude least
squares, ``h(z, mu) = mean_{tau,f} (sqrt(|S_z[tau,f]|^2 + mu^2) - sqrt(A[tau,f]))^2``,
minimized by trust-region Cauchy steps with block-stochastic gradients and a
mu-decrease rule. It runs on the same
:class:`~invexapi.operators.LiftedRadarAmbiguity` grids as waveX, so an
incomplete ambiguity function is simply a sparser grid.

With ``g_p[n] = x[n] conj(x[n - p])``, the lag-domain data satisfy
``Y[p, l] = sum_m G_l[p, m] conj(x_l[m])``, where ``x_l[m] = x[m] conj(x[m + l])``
and ``G_l[p, m] = x_l[m - p]`` is circulant. Since ``sum_n diag(X, 0)[n] = ||x||^2``,
the initial eigenvector is scaled by the square root of the positive part of ``x_0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from invexapi.operators import LiftedRadarAmbiguity

from .initialization import lag_domain

__all__ = ["BaselineResult", "alternating_init", "trust_region", "trust_region_baseline"]


@dataclass
class BaselineResult:
    x: torch.Tensor
    x_init: torch.Tensor
    gradient_norms: List[float] = field(default_factory=list)
    mus: List[float] = field(default_factory=list)


def _integer_rows(op: LiftedRadarAmbiguity) -> torch.Tensor:
    rows = op.delays.round()
    if not torch.allclose(rows, op.delays) or ((rows < 0) | (rows >= op.n)).any():
        raise ValueError("the alternating initialization needs integer delays in [0, N)")
    return rows.long()


def _circulant(v: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Rows ``p`` of ``G[p, m] = v[m - p]``."""
    n = len(v)
    m = torch.arange(n, device=v.device)
    return v[(m[None, :] - rows[:, None].to(v.device)) % n]


def _diagonals(w: torch.Tensor) -> torch.Tensor:
    """``x_l[m] = w[m] conj(w[m + l])`` for every l: an (N, N) tensor indexed [l, m]."""
    n = len(w)
    m = torch.arange(n, device=w.device)
    return w[None, :] * w[(m[None, :] + m[:, None]) % n].conj()


def alternating_init(
    op: LiftedRadarAmbiguity,
    af: torch.Tensor,
    iters: int = 2,
    lamb: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Proximal least squares on ``y_l = G_l conj(x_l)``, alternated with ``G_l``.

    Starts from ``x_init[p] = v[p] e^{i theta_p}`` with ``v[p] = (1/K) sum_f A[p, f]``.
    Each of ``iters`` rounds solves, in closed form for every lag l,
    ``min_u ||y_l - G_l u||^2 + ||u - u_prev||^2 / (2 lamb)``, writes the solutions
    on the diagonals of ``X_0`` and rebuilds ``G_l`` from its leading eigenvector.
    """
    n = op.n
    rows = _integer_rows(op)
    Y = lag_domain(op, af).to(op.D.dtype)            # (T, N): column l is y_l over the kept delays
    v = torch.zeros(n, dtype=torch.float64, device=af.device)
    v[rows.to(af.device)] = af.mean(dim=-1)
    theta = 2 * torch.pi * torch.rand(n, dtype=torch.float64, generator=generator).to(af.device)
    w = v * torch.exp(1j * theta)
    x_l = _diagonals(w)                              # diagonals of x_init x_init^H
    model = x_l                                      # the diagonals G_l is built from
    eye = torch.eye(n, dtype=op.D.dtype, device=af.device)
    m = torch.arange(n, device=af.device)
    for _ in range(iters):
        G = torch.stack([_circulant(model[l], rows) for l in range(n)])        # (N_l, T, N)
        Gh = G.conj().transpose(-1, -2)
        B = Gh @ G + eye / (2 * lamb)
        e = (Gh @ Y.T.unsqueeze(-1)).squeeze(-1) + x_l.conj() / (2 * lamb)
        x_l = torch.linalg.solve(B, e).conj()                                  # (N_l, N)
        X0 = torch.zeros(n, n, dtype=op.D.dtype, device=af.device)
        for l in range(n):
            X0[m, (m + l) % n] = x_l[l]
        _, V = torch.linalg.eigh(0.5 * (X0 + X0.conj().T))
        w = V[:, -1]
        model = _diagonals(w)
    energy = x_l[0].real.clamp(min=0).sum()          # sum over the positive entries
    return energy.sqrt() * w


def _smoothed_loss(op: LiftedRadarAmbiguity, b: torch.Tensor, x: torch.Tensor, rows, cols, mu: float):
    delayed = torch.fft.ifft(torch.fft.fft(x) * op.D[rows], dim=-1)
    s = ((x[None, :] * delayed.conj()) * op.E[cols]).sum(dim=-1)
    phi = torch.sqrt(s.real**2 + s.imag**2 + mu**2)
    return ((phi - b[rows, cols]) ** 2).mean()


def trust_region(
    op: LiftedRadarAmbiguity,
    af: torch.Tensor,
    x0: torch.Tensor,
    max_iter: int = 20000,
    batch: Optional[int] = None,
    alpha: float = 0.6,
    gamma: float = 0.1,
    gamma1: float = 0.1,
    mu0: float = 65.0,
    eps: float = 1e-10,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, List[float], List[float]]:
    """Trust-region Cauchy steps ``x <- x - alpha mu d / ||d||``, with ``d`` the Wirtinger
    gradient of ``h(z, mu)`` averaged over a random minibatch of ``batch`` (default N)
    samples, so that its expectation is the full gradient;
    ``mu <- gamma1 mu`` whenever ``||d|| < gamma mu``. Stops when ``||d||^2 < eps``.
    """
    b = af.clamp(min=0.0).sqrt()
    T, K = b.shape
    batch = op.n if batch is None else batch
    x = x0.detach().clone().to(op.D.dtype)
    mu = mu0
    norms, mus = [], []
    for _ in range(max_iter):
        idx = torch.randperm(T * K, generator=generator)[:batch].to(b.device)
        rows, cols = idx // K, idx % K
        xr = x.clone().requires_grad_(True)
        _smoothed_loss(op, b, xr, rows, cols, mu).backward()
        d = 0.5 * xr.grad      # torch returns 2 dh/dz-bar
        dn = torch.linalg.vector_norm(d).item()
        norms.append(dn)
        mus.append(mu)
        if dn**2 < eps or dn == 0.0:
            break
        x = x - alpha * mu * d / dn
        if dn < gamma * mu:
            mu *= gamma1
    return x.detach(), norms, mus


def trust_region_baseline(
    op: LiftedRadarAmbiguity,
    af: torch.Tensor,
    init_iters: int = 2,
    lamb: float = 1.0,
    generator: Optional[torch.Generator] = None,
    **trust_region_kwargs,
) -> BaselineResult:
    """:func:`alternating_init` then :func:`trust_region` on the ambiguity function ``af``."""
    x_init = alternating_init(op, af, iters=init_iters, lamb=lamb, generator=generator)
    x, norms, mus = trust_region(op, af, x_init, generator=generator, **trust_region_kwargs)
    return BaselineResult(x=x, x_init=x_init, gradient_norms=norms, mus=mus)
