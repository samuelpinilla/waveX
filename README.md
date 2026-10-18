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

The pulse is B-band-limited with B ≤ N/2: its spectrum vanishes outside B
contiguous DFT bins. This is the condition under which the AF determines almost
every pulse up to the trivial ambiguities. Modulation is a trivial ambiguity, so
only the band's width matters and waveX works in the centred band of B = N/2 bins:
`X = Q W Qᴴ` with `Q` the orthonormal in-band DFT columns and `W` Hermitian, B×B.
The band also removes the residue-class phases `x[n] ↦ e^{iφ_{n mod L}} x[n]`, which
leave the AF unchanged when only the delays `0, L, 2L, …` are measured and L
divides N: they mix copies of `x` modulated by multiples of N/L bins, which leave
the band. It also keeps decimated problems determined: an in-band W has B² real
unknowns instead of N².

## Algorithm

1. **Spectral initialization.** With `A(W) = B(Q W Qᴴ)` the in-band lifted operator
   and `b = √A`, from a random start `x = Qc` the initialization repeats the
   spectral step

   ```
   y = A(c cᴴ),      c ← lead( c cᴴ + A⁺( b ⊙ sgn(y) − y ) )
   ```

   where `lead(S) = √λ₁·v₁` is the leading eigenpair and `A⁺` is applied by a few
   conjugate-gradient steps from zero (the minimum-norm Hermitian correction). On
   the full grid `AᴴA = N·I` and the step is the back-projection
   `lead(Aᴴ(b ⊙ sgn(y)))`; on decimated grids the correction keeps the true pulse a
   fixed point even where `A` has a null space. Written as two maps on lifted
   matrices, `P_M(X) = X + A⁺(b ⊙ sgn(AX) − AX)` (fit the magnitudes) and
   `P_S(X) = lead(X)lead(X)ᴴ` (best rank-one fit), the plain step `X ← P_S(P_M(X))`
   can stall at a fixed point away from the pulse. The initialization therefore
   runs the relaxed averaged reflections

   ```
   X ← (β/2)(R_S R_M + I)X + (1 − β)·P_M(X),      R = 2P − I,   β = 0.9,
   ```

   from a single random start (300 steps); they keep the pulse a fixed point but
   escape those stalls. The anchor `x₀` is the leading factor of `P_S(P_M(X))`,
   scaled to `‖x‖ = A[0,0]^{1/4}`.

2. **Convex program.** Starting from `W₀ = c₀c₀ᴴ`, `c₀ = Qᴴx₀` (after centring the
   anchor's spectrum by an integer modulation), with `b = √A`, solve

   ```
   min_W  ½‖ |B(Q W Qᴴ)| + s − b ‖²  +  μ·tr(W)    subject to  W ⪰ 0
   ```

   by FASTA (adaptive forward–backward splitting) in three rounds. On the PSD cone
   the trace is the nuclear norm. Between rounds the slack is refreshed to
   `s = max(b − |B(Q W Qᴴ)|, 0)` and μ shrinks tenfold, starting from
   `μ = 10⁻²·‖eig(Aᴴb)‖₁`. The estimate is `Q·√λ₁v₁`, from the leading eigenpair of `W`.

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
python wavex/recover_pulse.py --bandwidth 128   # full space instead of the N/2 band
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
