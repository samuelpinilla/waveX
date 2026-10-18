"""waveX convex stage: recovery of x from its narrow-band ambiguity function.

The samples are ``A[tau, f] = |Tr(B_{tau,f} X)|^2`` at ``X = x x^H``. Started from
the anchor ``X0 = x0 x0^H``, the convex stage solves, in a penalized, slack-relaxed
form::

    min_Z  0.5*|| |B(Z)| + s - b ||^2  +  mu*tr(Z)  +  i_PSD(Z)

with ``b = sqrt(A)``, ``B`` the lifted measurement operator and the slack ``s``
refreshed between FASTA rounds as ``s = max(b - |B(Z)|, 0)``. On the PSD cone the
trace is the nuclear norm, the convex surrogate of rank. Each building block is
an invexapi object: ``LiftedRadarAmbiguity`` (B), ``SlackAmplitudeLeastSquares``
(smooth term), ``PSDTracePenalty(mu)`` (proximal term) and ``FASTA``. The rank-1 estimate ``sqrt(lambda_1) v_1`` is then
polished by ``LevenbergMarquardt`` on the signal-domain ``RadarAmbiguity`` operator.
With ``bandwidth`` set, the program runs over ``X = Q W Q^H`` with ``Q`` the centred
in-band basis of :mod:`wavex.band` (``W`` Hermitian, ``B x B``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from invexapi import SlackAmplitudeLeastSquares, PSDTracePenalty
from invexapi.operators import LiftedRadarAmbiguity, LinearOperator
from invexapi.optim import FASTA, LevenbergMarquardt

__all__ = ["WaveXResult", "trace_weight", "leading_component", "refine", "wavex"]


@dataclass
class WaveXResult:
    x: torch.Tensor
    Z: torch.Tensor
    mu: float
    x_convex: Optional[torch.Tensor] = None
    rounds: List[dict] = field(default_factory=list)
    refine_history: List[float] = field(default_factory=list)

    @property
    def residuals(self) -> List[float]:
        return [r for info in self.rounds for r in info["residuals"]]


def trace_weight(op: LinearOperator, b: torch.Tensor, scale: float) -> float:
    """``mu = scale * ||eig(B^*(b))||_1``: the trace weight, scaled to the data."""
    return scale * torch.linalg.eigvals(op.adjoint(b)).abs().sum().item()


def leading_component(Z: torch.Tensor) -> torch.Tensor:
    """``sqrt(lambda_1) v_1`` of the Hermitian part of ``Z`` ."""
    d, V = torch.linalg.eigh(0.5 * (Z + Z.conj().T))
    return torch.sqrt(d[-1].clamp(min=0.0)) * V[:, -1]


def refine(
    op: LiftedRadarAmbiguity, af: torch.Tensor, x: torch.Tensor, max_iter: int = 150, tol: float = 1e-15
) -> tuple[torch.Tensor, List[float]]:
    """Polish ``x`` by Levenberg-Marquardt on the intensity misfit ``A(x) - A``.

    The lifted convex stage only needs to land in the basin of this local
    problem (roughly 30% relative error); the polish then fits the data to the
    level the data allow; ``tol`` stops it once the relative misfit drops below.
    """
    solver = LevenbergMarquardt(op.signal_operator(), af, max_iter=max_iter, tol=tol)
    return solver.run(x)


def wavex(
    op: LiftedRadarAmbiguity,
    af: torch.Tensor,
    x0: torch.Tensor,
    outer_iters: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    mu: Optional[float] = None,
    mu_scale: float = 1e-2,
    mu_decay: float = 0.1,
    slack: str = "refresh",
    bandwidth: Optional[int] = None,
    refine_iters: int = 150,
    refine_tol: float = 1e-15,
    verbose: int = 0,
    generator: Optional[torch.Generator] = None,
) -> WaveXResult:
    """Recover ``x`` (up to trivial ambiguities) from ``af = op.ambiguity(x)``.

    ``x0`` is the anchor ``X0 = x0 x0^H`` that starts the convex stage, normally
    from :func:`wavex.initialization.spectral_init`. ``mu`` defaults to
    ``mu_scale * ||eig(B^*(b))||_1`` (:func:`trace_weight`). Each of ``outer_iters`` rounds
    runs FASTA on the lifted problem; between rounds ``mu`` is multiplied by
    ``mu_decay`` (continuation: a fixed mu biases the minimizer by O(mu)) and the
    slack is refreshed as ``max(b - |B(Z)|, 0)`` (``slack="refresh"``) or kept
    at zero (``slack="none"``). ``bandwidth`` restricts the program to the centred
    band of that many bins (the anchor is first modulated so that its strongest
    window is centred, a trivial ambiguity). ``refine_iters > 0`` then
    polishes the rank-1 estimate with :func:`refine` (stopping at ``refine_tol``).
    """
    if slack not in ("refresh", "none"):
        raise ValueError(f"slack must be 'refresh' or 'none', got {slack!r}")
    from .band import BandLimitedLift

    lift = op if bandwidth is None else BandLimitedLift(op, bandwidth)
    b = torch.sqrt(af.clamp(min=0.0))
    if mu is None:
        mu = trace_weight(lift, b, mu_scale)

    c0 = x0 if bandwidth is None else lift.coefficients(x0)
    Z = torch.outer(c0, c0.conj())
    s = torch.zeros_like(b)
    rounds = []
    mu_r = mu
    for r in range(outer_iters):
        solver = FASTA(
            SlackAmplitudeLeastSquares(b, slack=s),
            PSDTracePenalty(lamb=mu_r),
            A=lift,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            generator=generator,
        )
        Z, _ = solver.run(Z)
        rounds.append(solver.info)
        if slack == "refresh":
            s = torch.clamp(b - lift.apply(Z).abs(), min=0.0)
        mu_r *= mu_decay
        if verbose:
            print(f"waveX round {r + 1}/{outer_iters}: {solver.info['iterations']} FASTA iterations")

    x_convex = leading_component(Z) if bandwidth is None else lift.signal(leading_component(Z))
    x_hat, history = x_convex, []
    if refine_iters > 0:
        x_hat, history = refine(op, af, x_convex, max_iter=refine_iters, tol=refine_tol)
    return WaveXResult(x=x_hat, Z=Z, mu=mu, x_convex=x_convex, rounds=rounds, refine_history=history)

