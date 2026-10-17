"""Spectral initialization x^(0) of waveX for the narrow-band ambiguity function
``A[tau, f] = |Tr(B_{tau,f} x x^H)|^2``."""

from __future__ import annotations

import math

import torch

from invexapi.operators import LiftedRadarAmbiguity

__all__ = ["lag_domain", "estimate_norm", "spectral_matrix", "spectral_init"]


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


def spectral_init(op: LiftedRadarAmbiguity, af: torch.Tensor, keep_fraction: float = 1.0 / 6.0) -> torch.Tensor:
    """Leading eigenvector of :func:`spectral_matrix`, scaled by :func:`estimate_norm`."""
    _, V = torch.linalg.eigh(spectral_matrix(op, af, keep_fraction))
    return estimate_norm(op, af) * V[:, -1]
