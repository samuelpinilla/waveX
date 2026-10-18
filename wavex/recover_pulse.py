"""Recover a radar pulse from its noiseless narrow-band ambiguity function.

Loads a pulse from ``pulse_set.mat`` (pulse 3 by default), simulates its
ambiguity function on every L-th integer delay and every Doppler bin, builds the
spectral initialization and runs waveX - FASTA on the lifted convex program with
three slack rounds and mu continuation - then polishes the estimate by
Levenberg-Marquardt. Relative errors are dist(x, q)/||x|| up to the trivial
ambiguities, reported with a continuous time shift (default) and with integer
shifts only.

    python wavex/recover_pulse.py                                 # full AF, continuation, polish
    python wavex/recover_pulse.py --mu-decay 1 --refine-iters 0   # convex stage alone, fixed mu
    python wavex/recover_pulse.py --pulse-index 10 --decimation 3 --show
    python wavex/recover_pulse.py --baseline                      # also run the nonconvex baseline

Expect invexapi's "no convex/invex certificate" warnings: the amplitude loss is
nonconvex.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from invexapi.operators import LiftedRadarAmbiguity, integer_grid
from wavex import DEFAULT_PULSES, align, load_pulse, spectral_init, trust_region_baseline, wavex
from wavex.plotting import plot_comparison, plot_convergence


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=DEFAULT_PULSES, help="pulse_set.mat path")
    p.add_argument("--pulse-index", type=int, default=2, help="0-based row of pulse_set (2 = pulse 3)")
    p.add_argument("--decimation", type=int, default=1, help="L: keep every L-th integer delay")
    p.add_argument("--doppler-decimation", type=int, default=1, help="keep every L-th Doppler bin")
    p.add_argument("--bandwidth", type=int, default=None, help="restrict init and convex stage to this many centred bins")
    p.add_argument("--init-iters", type=int, default=300, help="spectral initialization steps")
    p.add_argument("--outer-iters", type=int, default=3, help="slack rounds")
    p.add_argument("--max-iter", type=int, default=1000, help="FASTA iterations per round")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--mu-scale", type=float, default=1e-2, help="mu = scale * ||eig(B^*(b))||_1")
    p.add_argument("--mu-decay", type=float, default=0.1, help="mu continuation factor per round (1 = fixed)")
    p.add_argument("--slack", choices=("refresh", "none"), default="refresh")
    p.add_argument("--refine-iters", type=int, default=150, help="Levenberg-Marquardt polish (0 = off)")
    p.add_argument("--baseline", action="store_true", help="also run the nonconvex baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("runs/recover_pulse"), help="figure directory")
    p.add_argument("--show", action="store_true", help="show figures interactively")
    p.add_argument("--verbose", type=int, default=1, help="0 quiet, 1 per round, 2 per iteration")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    x = load_pulse(args.pulse_index, args.data, device=device)
    n = x.shape[0]
    op = LiftedRadarAmbiguity(
        n, integer_grid(n, args.decimation), integer_grid(n, args.doppler_decimation), device=device
    )
    af = op.ambiguity(x)
    print(f"pulse {args.pulse_index}: N = {n}, {op.shape[0]} delays x {op.shape[1]} Doppler bins, m = {af.numel()}")

    x0 = spectral_init(op, af, bandwidth=args.bandwidth, iters=args.init_iters, generator=generator)
    xs = x.cpu().numpy()
    init = align(xs, x0.cpu().numpy())
    print(f"init error           {init.relative_error:.3e}")

    start = time.perf_counter()
    result = wavex(
        op, af, x0,
        outer_iters=args.outer_iters, max_iter=args.max_iter, tol=args.tol,
        mu_scale=args.mu_scale, mu_decay=args.mu_decay, slack=args.slack, bandwidth=args.bandwidth, refine_iters=args.refine_iters,
        verbose=max(args.verbose - 1, 0), generator=generator,
    )
    if args.verbose:
        for r, info in enumerate(result.rounds, start=1):
            print(f"  round {r}: {info['iterations']} iterations, final residual {info['residuals'][-1]:.3e}")
    convex = align(xs, result.x_convex.cpu().numpy())
    final = align(xs, result.x.cpu().numpy())
    strict = align(xs, result.x.cpu().numpy(), continuous_shift=False)
    print(f"mu = {result.mu:.3e}, solve time {time.perf_counter() - start:.1f}s")
    print(f"convex-stage error   {convex.relative_error:.3e}")
    if result.refine_history:
        print(f"refined error        {final.relative_error:.3e}  (misfit {result.refine_history[-1]:.1e}, "
              f"time shift {final.a:.4f} samples)")
    print(f"integer-shift error  {strict.relative_error:.3e}")

    if args.baseline:
        start = time.perf_counter()
        base = trust_region_baseline(op, af, lamb=10.0, generator=torch.Generator().manual_seed(args.seed))
        print(f"baseline: init {align(xs, base.x_init.cpu().numpy()).relative_error:.3e}, "
              f"final {align(xs, base.x.cpu().numpy()).relative_error:.3e} "
              f"({len(base.gradient_norms)} iterations, {time.perf_counter() - start:.1f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    plot_comparison(init.z, x0.cpu().numpy(), "initial point", args.out / "initial_point.png")
    plot_comparison(final.z, result.x.cpu().numpy(), "reconstruction", args.out / "reconstruction.png")
    plot_convergence(result.rounds, args.out / "convergence.png")
    print(f"figures saved to {args.out}/")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
