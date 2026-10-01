"""
Motif validation: how often each layer, role, and flag is right on XOR, common-driver, and chain networks.

Example::

    python validation/motifs.py --samples 3000 --reps 10
"""

import argparse

from infoflow.benchmarks import motif_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ds = motif_validation(args.samples, args.reps, prng=args.seed)
    for check, rate in zip(ds.check.values, ds["rate"].values, strict=True):
        print(f"{rate:6.2f}  {check}")


if __name__ == "__main__":
    main()
