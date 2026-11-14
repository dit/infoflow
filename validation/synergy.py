"""
Synergy validation: jointly informative parents and lag-0 explanation families on the mechanisms of
Hourican & Quax (2026), and synthetic modules planted beside a fixed background.

Example::

    python validation/synergy.py --samples 4000 --reps 10 --planted
"""

import argparse

from infoflow import benchmarks, datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planted", action="store_true", help="also plant modules beside a noisy rule-90 cellular automaton"
    )
    args = parser.parse_args()
    ds = benchmarks.synergy_validation(args.samples, args.reps, prng=args.seed)
    for check in ds.check.values:
        row = ds.sel(check=check)
        print(f"{float(row.rate):5.2f}  [{float(row.ci_low):.2f}, {float(row.ci_high):.2f}]  {check}", flush=True)
    if args.planted:
        background = datasets.noisy_eca(args.samples, n_cells=6, seed=args.seed)
        planted = benchmarks.planted_module_validation(background, n_reps=args.reps, prng=args.seed)
        for module in planted.module.values:
            row = planted.sel(module=module)
            low, high = (float(v) for v in row.recovered_ci.values)
            print(
                f"{float(row.recovered):5.2f}  [{low:.2f}, {high:.2f}]  exact {float(row.exact):.2f}  "
                f"false parents {float(row.false_parents):.2f}  planted {module}",
                flush=True,
            )


if __name__ == "__main__":
    main()
