import numpy as np
import pytest
import torch

from invexapi.operators import LiftedRadarAmbiguity, RadarAmbiguity, check_adjoint, integer_grid
from wavex import (
    alternating_init, align, estimate_norm, lag_domain, leading_component, load_pulses,
    refine, spectral_init, trust_region, trust_region_baseline, wavex,
)
from wavex.distance import fractional_shift


def _pulse():
    return load_pulses()[2]


def _af_reference(x, delays, dopplers):
    """Brute-force S[p, k] = sum_n x[n] conj(x[n-p]) e^{-2 pi i n k/N}, whose magnitude squared is the AF."""
    n = np.arange(len(x))
    return np.array([[np.sum(x * np.conj(x[(n - p) % len(x)]) * np.exp(-2j * np.pi * n * k / len(x)))
                      for k in dopplers] for p in delays])


def test_operator_matches_the_af_definition_on_integer_grids():
    x = _pulse()
    delays, dopplers = integer_grid(len(x), 3), integer_grid(len(x), 2)
    op = LiftedRadarAmbiguity(len(x), delays, dopplers)
    xt = torch.as_tensor(x)
    ref = _af_reference(x, delays.long().numpy(), dopplers.long().numpy())
    assert np.allclose(op.apply(torch.outer(xt, xt.conj())).numpy(), ref, atol=1e-12)
    assert np.allclose(op.measure(xt).numpy(), ref, atol=1e-12)
    assert np.allclose(op.signal_operator()(xt).numpy(), np.abs(ref) ** 2, atol=1e-12)
    assert isinstance(op.signal_operator(), RadarAmbiguity)


def test_operator_adjoint_and_full_grid_inverse():
    n = 32
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(n, n, dtype=torch.complex128, generator=gen)
    check_adjoint(LiftedRadarAmbiguity(n), x=X)
    check_adjoint(LiftedRadarAmbiguity(n, [-2.5, 0.3, 7.25], [0.5, 3.0, 11.7, 30.1]), x=X)
    op = LiftedRadarAmbiguity(n)
    assert torch.allclose(op.adjoint(op.apply(X)), n * X, atol=1e-10)


def test_trivial_ambiguities_leave_the_af_unchanged():
    x = _pulse()
    n = np.arange(len(x))
    op = LiftedRadarAmbiguity(len(x), np.arange(0, 128, 0.5)[:60], np.linspace(0.0, 127.0, 90))
    af = op.ambiguity(torch.as_tensor(x))
    same = [
        np.exp(0.7j) * x,
        x[(-n) % len(x)],                          # reflection without conjugation
        np.roll(x, 9),
        fractional_shift(x, 2.4),                  # Fourier-domain delay: fractional shifts too
        x * np.exp(2j * np.pi * 5 * n / len(x)),
    ]
    for q in same:
        assert torch.allclose(op.ambiguity(torch.as_tensor(q)), af, atol=1e-10 * af.max())
    # conj(x[-n]) maps A[tau, f] to A[tau, -f], so it is not an ambiguity
    conj_refl = op.ambiguity(torch.as_tensor(np.conj(x[(-n) % len(x)])))
    assert (conj_refl - af).abs().max() > 1e-2 * af.max()


@pytest.mark.parametrize("eps,a,b,beta", [(1, 0, 0.0, 0.0), (1, 5, 0.3, 1.0), (-1, 17, 2.0, -2.0)])
def test_align_inverts_every_trivial_ambiguity(eps, a, b, beta):
    x = _pulse()
    n = np.arange(len(x))
    q = np.exp(1j * beta) * np.exp(1j * b * n) * x[(eps * n - a) % len(x)]
    result = align(x, q)
    assert result.relative_error < 1e-6
    assert result.eps == eps and result.a == pytest.approx(a, abs=1e-6)


def test_norm_estimate_is_exact_when_the_origin_is_sampled():
    x = torch.as_tensor(_pulse()) * 3.0
    op = LiftedRadarAmbiguity(len(x), integer_grid(len(x), 4))
    assert estimate_norm(op, op.ambiguity(x)) == pytest.approx(3.0, rel=1e-10)


def test_lag_domain_is_the_autocorrelation_of_g():
    x = _pulse()
    op = LiftedRadarAmbiguity(len(x))
    Y = lag_domain(op, op.ambiguity(torch.as_tensor(x))).numpy()
    g = x * np.conj(np.roll(x, 3))                 # g_3[n] = x[n] conj(x[n-3])
    auto = np.array([np.sum(np.roll(g, -l) * np.conj(g)) for l in range(len(x))])
    assert np.allclose(Y[3], auto.real, atol=1e-12)


def test_leading_component_of_rank_one_matrix():
    x = torch.as_tensor(_pulse()) * 2.0
    v = leading_component(torch.outer(x, x.conj()))
    assert align(x.numpy(), v.numpy()).relative_error < 1e-8


def test_spectral_init_beats_a_random_start():
    x = torch.as_tensor(_pulse())
    op = LiftedRadarAmbiguity(len(x))
    af = op.ambiguity(x)
    x0 = spectral_init(op, af)
    assert x0.norm().item() == pytest.approx(1.0, rel=1e-8)
    assert align(x.numpy(), x0.numpy()).relative_error < 0.8


def test_refine_converges_from_a_nearby_start():
    x = torch.as_tensor(_pulse())
    op = LiftedRadarAmbiguity(len(x))
    d = torch.randn(len(x), dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    with pytest.warns(UserWarning):
        q, history = refine(op, op.ambiguity(x), x + 0.05 * d / d.norm(), max_iter=20)
    assert history[-1] < 1e-7
    assert align(x.numpy(), q.numpy()).relative_error < 1e-4


def test_wavex_pipeline_improves_on_its_initialization():
    x = torch.as_tensor(_pulse())
    op = LiftedRadarAmbiguity(len(x), integer_grid(len(x), 2))
    af = op.ambiguity(x)
    x0 = spectral_init(op, af)
    init_err = align(x.numpy(), x0.numpy()).relative_error
    with pytest.warns(UserWarning, match="certificate"):
        result = wavex(op, af, x0, outer_iters=2, max_iter=60, refine_iters=10,
                       generator=torch.Generator().manual_seed(0))
    assert align(x.numpy(), result.x_convex.numpy()).relative_error < init_err
    assert align(x.numpy(), result.x.numpy()).relative_error <= align(
        x.numpy(), result.x_convex.numpy()).relative_error
    with pytest.raises(ValueError):
        wavex(op, af, x0, slack="bogus")


def test_trust_region_baseline_runs_and_improves():
    x = torch.as_tensor(_pulse())
    op = LiftedRadarAmbiguity(len(x))
    af = op.ambiguity(x)
    g = torch.Generator().manual_seed(0)
    x_init = alternating_init(op, af, iters=2, lamb=10.0, generator=g)
    init_err = align(x.numpy(), x_init.numpy()).relative_error
    xr, norms, mus = trust_region(op, af, x_init, max_iter=3000, generator=g)
    assert align(x.numpy(), xr.numpy()).relative_error < init_err
    assert all(m1 >= m2 for m1, m2 in zip(mus, mus[1:]))   # mu never increases
    with pytest.raises(ValueError):
        trust_region_baseline(LiftedRadarAmbiguity(len(x), [0.5, 1.0]), af[:2])



@pytest.mark.parametrize("L,ambiguous", [(2, True), (4, True), (8, True), (3, False), (5, False)])
def test_decimated_delays_hide_residue_class_phases_iff_l_divides_n(L, ambiguous):
    """Keeping only delays tau = 0 mod L never mixes the classes n mod L when L | N,
    so an independent phase per class leaves the decimated AF unchanged."""
    x = torch.as_tensor(_pulse())
    n = torch.arange(len(x))
    op = LiftedRadarAmbiguity(len(x), integer_grid(len(x), L))
    af = op.ambiguity(x)
    phases = torch.tensor([0.0, 0.9, 2.1, -1.3, 0.4, 1.7, -2.4, 2.8])[n % L]
    q = x * torch.exp(1j * phases)
    change = ((op.ambiguity(q) - af).abs().max() / af.max()).item()
    assert align(x.numpy(), q.numpy()).relative_error > 0.1      # genuinely different signals
    assert (change < 1e-6) == ambiguous
