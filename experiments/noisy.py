"""Noisy, complete ambiguity functions: Gaussian noise on A at a given SNR.

SNR = 10 log10(||A||_F^2 / ||e||^2), swept over 10, 20, 30 and 40 dB.

    python experiments/noisy.py --pulses 0:100:5
"""

from pathlib import Path

from common import Scenario, arg_parser, pulse_range, run, summarize

if __name__ == "__main__":
    args = arg_parser(__doc__.split("\n\n")[0], "0:100:5").parse_args()
    out = args.out or Path("runs/noisy/results.jsonl")
    snrs = (10.0, 20.0, 30.0, 40.0)
    trials = [(p, Scenario(f"snr{int(s)}dB", snr_db=s, params=dict(snr_db=s)), args.seed)
              for s in snrs for p in pulse_range(args.pulses)]
    rows = run(trials, out, workers=args.workers, baseline=not args.no_baseline)
    table = summarize(rows)
    (out.parent / "summary.md").write_text(table + "\n")
    print(table)
