"""Shared runner for the waveX experiments: one (pulse, scenario) trial per process.

A trial simulates the ambiguity function of a pulse on the scenario's grids
(optionally with additive Gaussian noise at a given SNR), then runs waveX and,
unless disabled, the nonconvex baseline. Relative errors are dist(x, q)/||x||
up to the trivial ambiguities, with a continuous time shift. Results are written as JSON lines so long runs
can be inspected while they progress.
"""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# waveX defaults used by every experiment (see README, "Choosing the defaults")
WAVEX_DEFAULTS = dict(outer_iters=3, max_iter=1000, tol=1e-5, mu_decay=0.1, refine_iters=30)
BASELINE_LAMBDAS = (1.0, 10.0, 100.0)
SUCCESS = 1e-3


@dataclass
class Scenario:
    """Measurement grids and noise level of one experimental condition."""

    name: str
    delays: Optional[List[float]] = None        # None = all integer delays 0..N-1
    dopplers: Optional[List[float]] = None      # None = all integer Doppler bins 0..N-1
    snr_db: Optional[float] = None              # None = noiseless
    params: dict = field(default_factory=dict)  # free-form labels for plotting (e.g. removed fraction)


def add_noise(af, snr_db: float, generator):
    """``A + e`` with Gaussian ``e`` scaled so that ``10 log10(||A||_F^2 / ||e||^2) = snr_db``."""
    import torch

    e = torch.randn(af.shape, dtype=af.dtype, generator=generator).to(af.device)
    e = e * af.norm() / e.norm() * 10 ** (-snr_db / 20)
    return af + e


def trial(pulse: int, scenario: Scenario, seed: int = 0, baseline: bool = True, wavex_kwargs: Optional[dict] = None) -> dict:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    from invexapi.operators import LiftedRadarAmbiguity
    from wavex import align, load_pulse, spectral_init, trust_region_baseline, wavex

    x = load_pulse(pulse)
    xs = x.numpy()
    n = len(x)
    op = LiftedRadarAmbiguity(n, scenario.delays, scenario.dopplers)
    af = op.ambiguity(x)
    if scenario.snr_db is not None:
        af = add_noise(af, scenario.snr_db, torch.Generator().manual_seed(10_000 + seed))
    out = dict(pulse=pulse, scenario=scenario.name, seed=seed, m=af.numel(), **scenario.params)

    start = time.perf_counter()
    x0 = spectral_init(op, af)
    result = wavex(op, af, x0, generator=torch.Generator().manual_seed(seed), **{**WAVEX_DEFAULTS, **(wavex_kwargs or {})})
    out.update(
        wavex_time=time.perf_counter() - start,
        init_err=align(xs, x0.numpy()).relative_error,
        convex_err=align(xs, result.x_convex.numpy()).relative_error,
        wavex_err=align(xs, result.x.numpy()).relative_error,
        wavex_misfit=result.refine_history[-1] if result.refine_history else None,
    )

    if baseline:
        # lambda is chosen per pulse against the true error over a small grid (oracle cross-validation)
        best = None
        start = time.perf_counter()
        for lamb in BASELINE_LAMBDAS:
            base = trust_region_baseline(op, af, lamb=lamb, generator=torch.Generator().manual_seed(seed))
            err = align(xs, base.x.numpy()).relative_error
            if best is None or err < best[0]:
                best = (err, lamb, align(xs, base.x_init.numpy()).relative_error, len(base.gradient_norms))
        out.update(baseline_err=best[0], baseline_lambda=best[1], baseline_init_err=best[2],
                   baseline_iters=best[3], baseline_time=time.perf_counter() - start)
    return out


def run(trials: Iterable[tuple], out_path: Path, workers: int = 10, baseline: bool = True, wavex_kwargs: Optional[dict] = None) -> List[dict]:
    """Run ``(pulse, scenario, seed)`` trials in parallel, appending JSON lines to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trials = list(trials)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, out_path.open("w") as fh:
        futures = [pool.submit(trial, p, s, seed, baseline, wavex_kwargs) for p, s, seed in trials]
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            msg = f"[{i}/{len(trials)}] pulse {row['pulse']:3d} {row['scenario']:>14s}: waveX {row['wavex_err']:.2e}"
            if "baseline_err" in row:
                msg += f"  baseline {row['baseline_err']:.2e}"
            print(msg, flush=True)
    return rows


def summarize(rows: List[dict], key: str = "scenario") -> str:
    """Median errors and success rates (error < 1e-3) per scenario, as a Markdown table."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    has_base = any("baseline_err" in r for r in rows)
    head = "| scenario | trials | init | convex | waveX | waveX success |"
    sep = "|---|---|---|---|---|---|"
    if has_base:
        head += " baseline | baseline success |"
        sep += "---|---|"
    lines = [head, sep]
    for name, rs in groups.items():
        med = lambda k: float(np.median([r[k] for r in rs]))
        rate = lambda k: sum(r[k] < SUCCESS for r in rs) / len(rs)
        line = (f"| {name} | {len(rs)} | {med('init_err'):.2e} | {med('convex_err'):.2e} | "
                f"{med('wavex_err'):.2e} | {rate('wavex_err'):.0%} |")
        if has_base:
            line += f" {med('baseline_err'):.2e} | {rate('baseline_err'):.0%} |"
        lines.append(line)
    return "\n".join(lines)


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arg_parser(description: str, default_pulses: str):
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--pulses", default=default_pulses, help="python range spec 'start:stop:step' over pulse_set rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p


def pulse_range(spec: str) -> List[int]:
    parts = [int(v) if v else None for v in spec.split(":")]
    return list(range(100))[slice(*parts)]
