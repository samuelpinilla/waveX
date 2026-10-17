"""Pulse-set benchmark: complete, noiseless ambiguity function.

    python experiments/benchmark.py                 # all 100 pulses, waveX + baseline
    python experiments/benchmark.py --pulses 0:20   # first 20 pulses
"""

from pathlib import Path

from common import Scenario, arg_parser, pulse_range, run, summarize

if __name__ == "__main__":
    args = arg_parser(__doc__.split("\n\n")[0], "0:100").parse_args()
    out = args.out or Path("runs/benchmark/results.jsonl")
    trials = [(p, Scenario("complete"), args.seed) for p in pulse_range(args.pulses)]
    rows = run(trials, out, workers=args.workers, baseline=not args.no_baseline)
    table = summarize(rows)
    (out.parent / "summary.md").write_text(table + "\n")
    print(table)
