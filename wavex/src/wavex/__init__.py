"""waveX: radar waveform recovery from its narrow-band ambiguity function.

The ambiguity function ``A[tau, f] = |sum_n x[n] conj(x[n - tau]) e^{-2 pi i n f / N}|^2``
is the squared magnitude of a linear measurement of the lifted matrix
``X = x x^H``: ``A[tau, f] = |Tr(B_{tau,f} X)|^2``. waveX recovers ``x`` in three
steps: a spectral initialization, an anchored convex program over positive
semidefinite matrices, and a local polish. :mod:`wavex.baseline` provides a
nonconvex method for comparison.
"""

from .baseline import BaselineResult, alternating_init, trust_region, trust_region_baseline
from .data import DEFAULT_PULSES, load_pulse, load_pulses
from .distance import Alignment, align, distance, relative_error
from .initialization import estimate_norm, lag_domain, spectral_init, spectral_matrix
from .solver import WaveXResult, leading_component, refine, trace_weight, wavex

__all__ = [
    "DEFAULT_PULSES",
    "load_pulse",
    "load_pulses",
    "Alignment",
    "align",
    "distance",
    "relative_error",
    "estimate_norm",
    "lag_domain",
    "spectral_init",
    "spectral_matrix",
    "WaveXResult",
    "leading_component",
    "refine",
    "trace_weight",
    "wavex",
    "BaselineResult",
    "alternating_init",
    "trust_region",
    "trust_region_baseline",
]
