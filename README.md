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
(fractional ones included) or the reflection `x[−n]`. A fractional modulation
leaves the cyclic AF unchanged only if its ramp wraps where the pulse is
negligible, so the distance reads `n` on a window that wraps at the pulse's
weakest stretch. Errors are measured as `dist(x, q)/‖x‖`, the distance minimized
over these trivial ambiguities.

When only the delays `0, L, 2L, …` are measured and L divides N, the residue-class
phases `x[n] ↦ e^{iφ_{n mod L}} x[n]` leave the AF unchanged as well. They mix
copies of `x` modulated by multiples of N/L bins, which leave the pulse band, so
for band-limited pulses waveX can work in a centred band of B bins: `X = Q W Qᴴ`
with `Q` the orthonormal in-band DFT columns and `W` Hermitian, B×B. Modulation is
a trivial ambiguity, so only the band's width matters.

## Algorithm

1. **Spectral initialization.** With `A(W) = B(Q W Qᴴ)` (`Q = I`, or the in-band
   basis) and `b = √A`, from a random start `x = Qc` the initialization repeats the
   spectral step

   ```
   y = A(c cᴴ),      c ← lead( c cᴴ + A⁺( b ⊙ sgn(y) − y ) )
   ```

   where `lead(S) = √λ₁·v₁` is the leading eigenpair and `A⁺` is applied by a few
   conjugate-gradient steps from zero (the minimum-norm Hermitian correction). On
   the full grid `AᴴA = N·I` and the step is the back-projection
   `lead(Aᴴ(b ⊙ sgn(y)))`; on decimated grids the correction keeps the true pulse a
   fixed point even where `A` has a null space. The iteration either reaches the
   pulse or stalls at a fixed point with a clearly higher misfit, so it is run from
   three random starts and the result with the lowest misfit `‖A(x₀) − A‖/‖A‖` is
   kept. It is scaled to `‖x‖ = A[0,0]^{1/4}` and is the anchor `x₀`.

2. **Convex program.** Starting from `Z = X₀ = x₀x₀ᴴ` (or `W = c₀c₀ᴴ` in the band),
   with `b = √A`, solve

   ```
   min_Z  ½‖ |B(Z)| + s − b ‖²  +  μ·tr(Z)    subject to  Z ⪰ 0
   ```

   by FASTA (adaptive forward–backward splitting) in three rounds. On the PSD cone
   the trace is the nuclear norm. Between rounds the slack is refreshed to
   `s = max(b − |B(Z)|, 0)` and μ shrinks tenfold, starting from
   `μ = 10⁻²·‖eig(Bᴴb)‖₁`. The estimate is `√λ₁·v₁`, the leading eigenpair of `Z`.

3. **Polish.** Levenberg–Marquardt on the intensity misfit `A(x) − A`, started from
   the convex estimate.

| Role | Object |
|---|---|
| Lifted AF `Z ↦ Tr(B_{τ,f} Z)` | `invexapi.operators.LiftedRadarAmbiguity` |
| Signal-domain AF `x ↦ A[τ, f]` | `invexapi.operators.RadarAmbiguity` |
| Data fidelity with slack | `invexapi.SlackAmplitudeLeastSquares` |
| In-band lift `W ↦ Tr(B_{τ,f} Q W Qᴴ)` | `wavex.BandLimitedLift` |
| Trace and PSD constraint | `invexapi.PSDTracePenalty(μ)` |
| Convex solver, polish | `invexapi.optim.FASTA`, `invexapi.optim.LevenbergMarquardt` |

## Install and use

```bash
pip install -e ../invexapi      # or wherever invexapi lives
pip install -e .
python wavex/recover_pulse.py                   # pulse 3, complete AF
python wavex/recover_pulse.py --decimation 3    # keep every 3rd delay
python wavex/recover_pulse.py --decimation 2 --bandwidth 64   # every 2nd delay, in band
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
    band.py              centred in-band lifted operator
    initialization.py    spectral initialization
    solver.py            convex program and polish
    distance.py          distance up to trivial ambiguities
    baseline.py          nonconvex baseline for comparison
    data.py, plotting.py
experiments/             benchmark, incomplete, noisy, plots
tests/test_wavex.py
```
