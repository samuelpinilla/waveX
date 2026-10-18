"""The pulse band: centred in-band lifted operator for band-limited pulses.

A modulation ``e^{ibn} x`` leaves the ambiguity function unchanged, so the band of
a pulse occupying ``B`` contiguous DFT bins can always be taken centred,
``k in [-B/2, B/2)``. Restricting ``x`` to that band removes the residue-class
phases ``x[n] -> e^{i phi_{n mod L}} x[n]`` that leave an ambiguity function
measured on the delays ``tau = 0, L, 2L, ...`` unchanged when ``L`` divides ``N``:
they mix copies of ``x`` modulated by multiples of ``N/L`` bins, whose spectra
leave a band of at most ``N/L`` bins entirely (and partly for wider bands).
"""

from __future__ import annotations

import numpy as np
import torch

from invexapi.operators import LinearOperator

__all__ = ["band_basis", "center_spectrum", "BandLimitedLift"]


def _centred_bins(n: int, bandwidth: int) -> np.ndarray:
    k = np.fft.fftfreq(n, d=1.0 / n)
    return np.flatnonzero((k >= -(bandwidth // 2)) & (k < bandwidth - bandwidth // 2))


def band_basis(n: int, bandwidth: int, dtype: torch.dtype = torch.complex128, device=None) -> torch.Tensor:
    """``Q[m, j] = e^{2 pi i m k_j / n} / sqrt(n)`` for the ``bandwidth`` centred bins ``k_j``."""
    if not 1 <= bandwidth <= n:
        raise ValueError(f"bandwidth must be in [1, {n}], got {bandwidth}")
    k = torch.as_tensor(np.fft.fftfreq(n, d=1.0 / n)[_centred_bins(n, bandwidth)], dtype=torch.float64)
    time = torch.arange(n, dtype=torch.float64)
    return (torch.exp(2j * torch.pi * torch.outer(time, k) / n) / n**0.5).to(dtype=dtype, device=device)


def center_spectrum(x: torch.Tensor, bandwidth: int) -> torch.Tensor:
    """``e^{-2 pi i s n / N} x``: the integer modulation that centres the strongest ``bandwidth``-bin window.

    A modulation is a trivial ambiguity, so this changes nothing but the
    representative; it lets a signal be projected onto the centred band.
    """
    n = x.shape[-1]
    power = (torch.fft.fft(x).abs() ** 2).cpu().numpy()
    cyclic = np.concatenate([power, power[: bandwidth - 1]])
    start = int(np.argmax(np.convolve(cyclic, np.ones(bandwidth), mode="valid")[:n]))
    shift = start + bandwidth // 2  # the window's centre bin goes to 0
    ramp = torch.exp(-2j * torch.pi * shift * torch.arange(n, dtype=torch.float64, device=x.device) / n)
    return x * ramp.to(x.dtype)


class BandLimitedLift(LinearOperator):
    """``W -> op.apply(Q W Q^H)``: the lifted measurement restricted to in-band signals.

    A band-limited ``x = Q c`` lifts to ``x x^H = Q (c c^H) Q^H``, so the convex
    variable becomes the ``B x B`` Hermitian matrix ``W``, with ``tr(Q W Q^H) =
    tr(W)`` and ``Q W Q^H`` PSD exactly when ``W`` is.
    """

    def __init__(self, op: LinearOperator, bandwidth: int):
        self.op = op
        self.bandwidth = bandwidth
        self.Q = band_basis(op.n, bandwidth, dtype=op.D.dtype, device=op.D.device)
        self.in_shape = (bandwidth, bandwidth)
        self.out_shape = op.out_shape

    @property
    def shape(self):
        return self.out_shape

    def coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """In-band coefficients ``c = Q^H x`` of ``x`` after centring its spectrum."""
        return self.Q.conj().T @ center_spectrum(x.to(self.Q.dtype), self.bandwidth)

    def signal(self, c: torch.Tensor) -> torch.Tensor:
        """``x = Q c``."""
        return self.Q @ c

    def apply(self, W: torch.Tensor) -> torch.Tensor:
        return self.op.apply(self.Q @ W.to(self.Q.dtype) @ self.Q.conj().T)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.Q.conj().T @ self.op.adjoint(y) @ self.Q
