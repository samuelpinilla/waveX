"""Spectral initialization x^(0) of waveX for the narrow-band ambiguity function
``A[tau, f] = |Tr(B_{tau,f} x x^H)|^2``.

With ``A`` the in-band lifted operator ``W -> B(Q W Q^H)`` (``Q`` the centred
in-band basis of :mod:`wavex.band`), the initialization alternates two steps
from a random start ``x = Q c``: give the measured magnitudes ``b = sqrt(A)`` the
phases of the current estimate, then correct the lift ``c c^H`` by the
minimum-norm Hermitian update that fits them, and take the leading eigenvector::

    y = A(c c^H),     c <- lead( c c^H + A^+( b * sgn(y) - y ) ).

Each step is a spectral step: the leading eigenpair of a Hermitian matrix built
linearly from the data. Run as relaxed averaged reflections between the two
steps (:func:`spectral_init`), one random start suffices. On the full integer grid ``A^*A = N I``, so the update
is the back-projection ``lead(A^*(b * sgn(y)))``; on decimated grids the
correction keeps the true pulse a fixed point even where ``A`` has a null space.
``A^+`` is applied by a few conjugate-gradient steps from zero.
:func:`lag_spectral_init` is a one-shot alternative built from the lag-domain
data.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from invexapi.operators import LiftedRadarAmbiguity, LinearOperator

from .band import BandLimitedLift
from .solver import leading_component

__all__ = [
    "lag_domain", "estimate_norm", "spectral_matrix", "lag_spectral_init",
    "min_norm_update", "spectral_init",
]


def lag_domain(op: LiftedRadarAmbiguity, af: torch.Tensor) -> torch.Tensor:
    """``Y[tau, l] = (1/K) sum_f A[tau, f] e^{2 pi i l f / N}``, l = 0..N-1.

    On the full integer Doppler grid this is the inverse DFT along f, and
    ``Y[tau, .]`` is the cyclic autocorrelation of ``g_tau[n] = x[n] conj(x[n - tau])``.
    """
    lags = torch.arange(op.n, dtype=torch.float64, device=af.device)
    W = torch.exp(2j * torch.pi * torch.outer(op.dopplers.to(af.device), lags) / op.n)
    return (af.to(W.dtype) @ W).real / len(op.dopplers)


def estimate_norm(op: LiftedRadarAmbiguity, af: torch.Tensor) -> float:
    """``||x||_2`` from the ambiguity function.

    ``S[0, 0] = ||x||^2``, so ``A[0, 0]^(1/4)`` is exact when the origin is
    sampled. Otherwise ``sum_{tau, l} Y[tau, l] = sum_tau sum_n |x[n]|^2 |x[n - tau]|^2``,
    which equals ``(T/N) ||x||^4`` only on average, is used as a fallback.
    """
    tau0 = (op.delays == 0).nonzero()
    f0 = (op.dopplers == 0).nonzero()
    if len(tau0) and len(f0):
        return af[tau0[0, 0], f0[0, 0]].clamp(min=0.0).item() ** 0.25
    total = lag_domain(op, af)[:, 0].sum().clamp(min=0.0).item()
    return (total * op.n / len(op.delays)) ** 0.25


def spectral_matrix(op: LiftedRadarAmbiguity, af: torch.Tensor, keep_fraction: float = 1.0 / 6.0) -> torch.Tensor:
    """``G_0 = Herm(sum_{(tau,l) in I} Y[tau,l] Phi_tau^H e_l e_l^T)``.

    ``I`` holds the ``floor(m * keep_fraction)`` largest lag-domain entries, after
    lag 0 is moved to the grid centre (a trivial shift ambiguity), and each lag is
    read as a sample index. A kept entry measures ``g_tau[l] = x^H (Phi_tau^H e_l e_l^T) x``,
    so ``G_0`` is the sum of these rank-one matrices weighted by ``Y``. It is formed
    in one matmul: ``G = ifft_rows(P * (conj(D)^T @ w))``.
    """
    n = op.n
    Y = torch.roll(lag_domain(op, af), n // 2, dims=-1)
    count = max(1, math.floor(Y.numel() * keep_fraction))
    flat = Y.reshape(-1)
    w = torch.zeros_like(flat)
    top = torch.topk(flat, count).indices
    w[top] = flat[top]
    w = w.reshape(Y.shape).to(op.D.dtype)
    G = torch.fft.ifft(op.P * (op.D.conj().T @ w), dim=0)
    return 0.5 * (G + G.conj().T)


def lag_spectral_init(op: LiftedRadarAmbiguity, af: torch.Tensor, keep_fraction: float = 1.0 / 6.0) -> torch.Tensor:
    """Leading eigenvector of :func:`spectral_matrix`, scaled by :func:`estimate_norm`."""
    _, V = torch.linalg.eigh(spectral_matrix(op, af, keep_fraction))
    return estimate_norm(op, af) * V[:, -1]


def _herm(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.conj().T)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real


def min_norm_update(lift: LinearOperator, r: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Approximately ``A^+ r`` over Hermitian matrices: CG from zero on ``herm(A^*A D) = herm(A^* r)``.

    Starting from zero keeps every iterate in the range of ``A^*``, so the null
    space of ``A`` is left untouched (the minimum-norm solution).
    """
    res = _herm(lift.adjoint(r))
    D = torch.zeros_like(res)
    p = res.clone()
    rr = _dot(res, res)
    for _ in range(iters):
        Ap = _herm(lift.adjoint(lift.apply(p)))
        pAp = _dot(p, Ap)
        if not torch.isfinite(pAp) or pAp <= 1e-30:
            break
        alpha = rr / pAp
        D = D + alpha * p
        res = res - alpha * Ap
        rr_new = _dot(res, res)
        if rr_new.sqrt() < 1e-12:
            break
        p = res + (rr_new / rr) * p
        rr = rr_new
    return D


def spectral_init(
    op: LiftedRadarAmbiguity,
    af: torch.Tensor,
    bandwidth: Optional[int] = None,
    iters: int = 300,
    beta: float = 0.9,
    cg_iters: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """The anchor ``x0``: ``iters`` relaxed spectral steps from one random start in the band.

    With ``P_M(X) = X + A^+(b * sgn(A X) - A X)`` (fit the magnitudes, keeping the
    current phases) and ``P_S(X) = lead(X) lead(X)^H`` (best rank-one PSD fit), the
    plain iteration ``X <- P_S(P_M(X))`` can stall at a fixed point away from the
    pulse. The relaxed averaged reflections
    ``X <- (beta/2) (R_S R_M + I) X + (1 - beta) P_M(X)``, with ``R = 2P - I``,
    keep the pulse a fixed point but escape those stalls; ``beta = 0`` is the plain
    iteration. The anchor is the leading factor of ``P_S(P_M(X))``.
    The pulse is taken ``bandwidth``-band-limited (default ``N // 2``, the widest
    band for which the AF determines almost every pulse; ``bandwidth = N`` is the
    full space); ``cg_iters`` CG steps apply ``A^+``
    (:func:`min_norm_update`). Start and result are scaled by :func:`estimate_norm`.
    """
    lift = BandLimitedLift(op, op.n // 2 if bandwidth is None else bandwidth)
    b = torch.sqrt(af.clamp(min=0.0)).to(op.D.dtype)
    norm = estimate_norm(op, af)

    def fit_magnitudes(X: torch.Tensor) -> torch.Tensor:
        y = lift.apply(X)
        return X + min_norm_update(lift, b * torch.sgn(y) - y, cg_iters)

    def rank_one(X: torch.Tensor) -> torch.Tensor:
        v = leading_component(X)
        return torch.outer(v, v.conj())

    c = torch.randn(lift.in_shape[0], dtype=op.D.dtype, device=op.D.device, generator=generator)
    c = norm * c / torch.linalg.vector_norm(c)
    X = torch.outer(c, c.conj())
    for _ in range(iters):
        pm = fit_magnitudes(X)
        rm = 2 * pm - X
        X = 0.5 * beta * (2 * rank_one(rm) - rm + X) + (1 - beta) * pm
    c = leading_component(fit_magnitudes(X))
    return norm * lift.signal(c / torch.linalg.vector_norm(c))
