"""Dependency-light worked example: seeded coordinates -> exact counts -> PNG.

Run: python examples/coordinates.py --out results/coordinates
"""
import argparse
import json
from pathlib import Path

import numpy as np

from synthonmap import Histogram, fit_bounds, coarsen_counts, display_ceiling, save_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/coordinates"))
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    if args.count < 1 or args.size < 1:
        parser.error("count and size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    def batches():
        # Each call resets the seed: bounds and count passes see identical draws.
        rng = np.random.default_rng(47)
        centers = np.array([[-2, 0], [0, 2], [2, -.5]])
        for start in range(0, args.count, 65536):
            n = min(65536, args.count - start)
            yield centers[rng.integers(3, size=n)] + rng.normal(size=(n, 2)) * [.55, .25]

    histogram = Histogram(fit_bounds(batches()), (args.size, args.size))
    for xy in batches():
        histogram.add(xy)
    assert histogram.accepted == args.count
    np.save(args.out / "counts.npy", histogram.counts)
    ceiling = display_ceiling(histogram.counts)
    for factor in [1, 8, 64]:
        level = coarsen_counts(histogram.counts, factor)
        save_png(args.out / f"density-{factor}x.png", level.counts, areas=level.areas, ceiling=ceiling)
    report = histogram.receipt() | dict(display_ceiling=ceiling, seed=47, synthetic=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
