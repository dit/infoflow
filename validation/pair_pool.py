"""
Pair-search coverage: XOR parents at random processes and lags, old pool (the first 20
candidates in lag order, no curtailment) against the default pool (up to max_pairs pairs,
curtailed), with null targets for false parents and time.

Example::

    python validation/pair_pool.py --processes 8 12 20 --reps 10
"""

import argparse
import time

import numpy as np

from infoflow.data import DiscreteData
from infoflow.embedding import Embedding
from infoflow.selection import SkeletonSettings, select_parents

SETTINGS = {
    "first 20": SkeletonSettings(max_pair_candidates=20, max_pairs=None, curtail=False),
    "max_pairs": SkeletonSettings(),
}


def simulate(n, n_processes, parents, flip, rng):
    """
    Independent fair bits; the last process is the noisy XOR of the (process, lag) parents.
    """
    x = rng.integers(0, 2, (n, n_processes))
    y = np.zeros(n, dtype=int)
    for p, lag in parents:
        y ^= np.roll(x[:, p], lag)
    x[:, -1] = y ^ (rng.random(n) < flip) if parents else x[:, -1]
    return DiscreteData.from_discrete(x)


def run(n_processes, reps, n, flip, seed):
    rows = []
    for kind in ("xor", "null"):
        for r in range(reps):
            rng = np.random.default_rng([seed, n_processes, r, kind == "xor"])
            if kind == "xor":
                a, b = rng.choice(n_processes - 1, 2, replace=False)
                parents = [(int(a), int(rng.integers(1, 5))), (int(b), int(rng.integers(1, 5)))]
            else:
                parents = []
            data = simulate(n, n_processes, parents, flip, rng)
            for name, settings in SETTINGS.items():
                start = time.perf_counter()
                sk = select_parents(data, n_processes - 1, Embedding(max_lag=1), settings=settings, prng=r)
                elapsed = time.perf_counter() - start
                found = set(sk.sources) if sk.significant else set()
                rows.append(
                    {
                        "processes": n_processes,
                        "kind": kind,
                        "settings": name,
                        "recovered": set(parents) <= found if parents else None,
                        "false": len(found - set(parents)),
                        "seconds": elapsed,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--processes", type=int, nargs="+", default=[8, 12, 20])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--flip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"{'processes':>9}  {'settings':>9}  {'XOR recovered':>13}  {'false/target':>12}  {'s/target':>8}")
    for P in args.processes:
        rows = run(P, args.reps, args.samples, args.flip, args.seed)
        for name in SETTINGS:
            xor = [r for r in rows if r["settings"] == name and r["kind"] == "xor"]
            null = [r for r in rows if r["settings"] == name and r["kind"] == "null"]
            recovered = sum(r["recovered"] for r in xor)
            false = np.mean([r["false"] for r in xor + null])
            seconds = np.mean([r["seconds"] for r in xor + null])
            print(f"{P:>9}  {name:>9}  {recovered:>6}/{len(xor):<6}  {false:>12.2f}  {seconds:>8.1f}")


if __name__ == "__main__":
    main()
