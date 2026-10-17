"""Incomplete ambiguity functions: removed delays or Doppler bins.

Scenarios (N = 128):

* ``delays/L``: keep every L-th integer delay, L = 1, 2, 3, 4, 5, 7, 8. Rows record
  whether L is coprime to N (``identifiable``).
* ``delays-mid50`` / ``doppler-mid50``: keep indices N/4 .. 3N/4 - 1 of the
  delay or Doppler axis.
* ``delays-centre50`` / ``doppler-centre50``: keep |tau| < N/4 or |f| < N/4 on the
  centred axis, i.e. indices 0 .. N/4 - 1 and 3N/4 .. N - 1.

    python experiments/incomplete.py --pulses 0:100:5
    python experiments/incomplete.py --only delays/5 delays/7 --out runs/incomplete/coprime.jsonl
    python experiments/incomplete.py --only delays-centre50 doppler-centre50 --out runs/incomplete/centre.jsonl
"""

import math
from pathlib import Path

from common import Scenario, arg_parser, pulse_range, run, summarize

N = 128


def scenarios():
    out = []
    for L in (1, 2, 3, 4, 5, 7, 8):
        kept = list(range(0, N, L))
        out.append(Scenario(f"delays/{L}", delays=[float(t) for t in kept],
                            params=dict(removed=1 - len(kept) / N, kind="uniform delays",
                                        identifiable=math.gcd(L, N) == 1)))
    middle = [float(t) for t in range(N // 4, 3 * N // 4)]
    out.append(Scenario("delays-mid50", delays=middle, params=dict(removed=0.5, kind="non-uniform delays")))
    out.append(Scenario("doppler-mid50", dopplers=middle, params=dict(removed=0.5, kind="non-uniform Doppler")))
    centre = [float(t) for t in list(range(N // 4)) + list(range(3 * N // 4, N))]
    out.append(Scenario("delays-centre50", delays=centre, params=dict(removed=0.5, kind="non-uniform delays")))
    out.append(Scenario("doppler-centre50", dopplers=centre, params=dict(removed=0.5, kind="non-uniform Doppler")))
    return out


if __name__ == "__main__":
    parser = arg_parser(__doc__.split("\n\n")[0], "0:100:5")
    parser.add_argument("--only", nargs="*", default=None, help="scenario names to run (default: all)")
    args = parser.parse_args()
    out = args.out or Path("runs/incomplete/results.jsonl")
    chosen = [s for s in scenarios() if args.only is None or s.name in args.only]
    trials = [(p, s, args.seed) for s in chosen for p in pulse_range(args.pulses)]
    rows = run(trials, out, workers=args.workers, baseline=not args.no_baseline)
    table = summarize(rows)
    (out.parent / f"{out.stem}_summary.md").write_text(table + "\n")
    print(table)
