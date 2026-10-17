# waveX

**Recover a radar waveform from its narrow-band ambiguity function by convex optimization.**

Given only the magnitude of the discrete narrow-band ambiguity function (AF) of a
pulse `x ∈ ℂᴺ`, waveX reconstructs the pulse, magnitude *and* phase, up to the
AF's trivial ambiguities. The numerical building blocks (operators, penalties and
solvers) come from [invexapi](https://github.com/samuelpinilla/invexapi).

## Problem and lifted model

The AF samples are

```
A[τ, f] = | Σₙ x[n] conj((Φ_τ x)[n]) e^{−2πinf/N} |²,      Φ_τ = Fᴴ diag(e^{−2πilτ/N}) F,
```

with `F` the DFT and the frequency index centred, `l ∈ [−N/2, N/2)`. The delay
`Φ_τ` is a Fourier-domain phase ramp. On integer grids it is the cyclic shift
`(Φ_τ x)[n] = x[n − τ]`, and it also allows fractional delays and Doppler.

The AF is quartic in `x`, but lifting `x` to `X = x xᴴ` makes every sample the
squared magnitude of a linear measurement:

```
A[τ, f] = |Tr(B_{τ,f} X)|²,      B_{τ,f} = Φ_τᴴ diag(e^{−2πinf/N}).
```

`Tr(B_{τ,f} X)` is the Doppler transform of the diagonal of `X Φ_τᴴ`, so the lifted
operator is matrix-free and costs O(N²(T + K)) for T delays and K Doppler bins. On
the full integer grid, `BᴴB = N·I`.

The AF does not change under a global phase, a modulation `e^{ibn}`, a time shift
(fractional ones included) or the reflection `x[−n]`. Errors are therefore
measured as `dist(x, q)/‖x‖`, the distance minimized over these trivial
ambiguities.

## Algorithm

1. **Spectral initialization.** The lag-domain data `Y[τ, l] = IDFT_f A[τ, f]` is
   the autocorrelation of `g_τ[n] = x[n] x̄[n − τ]`. After lag 0 is moved to the
   grid centre, the largest ⌊m/6⌋ entries are kept. Each kept entry is the
   quadratic form `g_τ[l] = xᴴ(Φ_τᴴ e_l e_lᵀ)x`, giving the spectral matrix

   ```
   G₀ = Herm( Σ_{(τ,l) kept} Y[τ, l] Φ_τᴴ e_l e_lᵀ ).
   ```

   Its leading eigenvector, scaled to `‖x‖ = A[0,0]^{1/4}`, is the anchor `x₀`.

2. **Anchored convex program.** With `b = √A` and `X₀ = x₀x₀ᴴ`, solve

   ```
   min_Z  ½‖ |B(Z)| + s − b ‖²  −  μ·Re⟨X₀, Z⟩  +  μ·tr(Z)    subject to  Z ⪰ 0
   ```

   by FASTA (adaptive forward–backward splitting) in three rounds. Between rounds
   the slack is refreshed to `s = max(b − |B(Z)|, 0)` and μ shrinks tenfold, starting
   from `μ = 10⁻⁵·‖eig(Bᴴb)‖₁`. The estimate is `√λ₁·v₁`, the leading eigenpair of `Z`.

3. **Polish.** Levenberg–Marquardt on the intensity misfit `A(x) − A`, started from
   the convex estimate.

| Role | Object |
|---|---|
| Lifted AF `Z ↦ Tr(B_{τ,f} Z)` | `invexapi.operators.LiftedRadarAmbiguity` |
| Signal-domain AF `x ↦ A[τ, f]` | `invexapi.operators.RadarAmbiguity` |
| Data fidelity with slack | `invexapi.SlackAmplitudeLeastSquares` |
| Anchor, trace and PSD constraint | `invexapi.LinearTilt(PSDTracePenalty(μ), μ·X₀)` |
| Convex solver, polish | `invexapi.optim.FASTA`, `invexapi.optim.LevenbergMarquardt` |

## Install and use

```bash
pip install -e ../invexapi      # or wherever invexapi lives
pip install -e .
python wavex/recover_pulse.py                   # pulse 3, complete AF
python wavex/recover_pulse.py --decimation 3    # keep every 3rd delay
```

```python
from invexapi.operators import LiftedRadarAmbiguity
from wavex import align, load_pulse, spectral_init, wavex

x = load_pulse(2)                       # ground truth, for scoring only
op = LiftedRadarAmbiguity(len(x))       # all integer delays and Doppler bins
af = op.ambiguity(x)                    # the measurements

result = wavex(op, af, spectral_init(op, af))
print(align(x.numpy(), result.x.numpy()).relative_error)
```

`experiments/` reproduces the numerical comparisons (benchmark, incomplete and noisy
AFs, figures). `pytest` runs the tests.

## Layout

```
wavex/
  recover_pulse.py       command-line driver
  data/pulse_set.mat     benchmark pulses (100 x 128, complex)
  src/wavex/
    initialization.py    spectral initialization
    solver.py            anchored convex program and polish
    distance.py          distance up to trivial ambiguities
    baseline.py          nonconvex baseline for comparison
    data.py, plotting.py
experiments/             benchmark, incomplete, noisy, plots
tests/test_wavex.py
```
