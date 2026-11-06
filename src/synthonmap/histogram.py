"""Streaming integer histograms, in image order (row zero is highest Y)."""
from dataclasses import dataclass
import operator

import numpy as np


def _shape(shape):
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    result = tuple(operator.index(v) for v in shape)
    if min(result) < 1:
        raise ValueError("histogram dimensions must be positive")
    return result


def _bounds(bounds):
    result = np.array(bounds, dtype=np.float64, copy=True)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError("bounds must be finite [[xmin, ymin], [xmax, ymax]]")
    if not np.all(result[1] > result[0]) or not np.isfinite(result[1] - result[0]).all():
        raise ValueError("bounds must have positive finite spans")
    return result


def _points(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    return xy


def fit_bounds(batches, *, padding=0.015, square=True):
    """One bounded-memory pass over coordinates, retaining every finite tail.

    Replay the same coordinate stream for accumulation. Nonfinite rows are
    ignored here and explicitly counted by Histogram.add. No quantile crop.
    """
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be nonnegative and finite")
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for batch in batches:
        xy = _points(batch)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            lo = np.minimum(lo, xy.min(axis=0))
            hi = np.maximum(hi, xy.max(axis=0))
    if not np.isfinite(lo).all():
        raise ValueError("cannot fit bounds without finite coordinates")
    span = hi - lo
    if not np.isfinite(span).all():
        raise ValueError("coordinate span is too large")
    # A constant coordinate still needs a nonzero plotting interval.
    span = np.maximum(span, np.maximum(np.abs(lo), 1) * 1e-6)
    if square:
        span[:] = span.max()
    half = span * (0.5 + padding)
    center = lo / 2 + hi / 2
    return _bounds([center - half, center + half])


class Histogram:
    """uint64 counts with O(batch + grid) memory, never O(all draws).

    Outside points are dropped, not clamped; exact outer edges are included.
    Inner edges follow floor((1-normalized_y)*height) in image coordinates,
    matching the original renderer. A supplied zeroed uint64 memmap allows
    the grid to live on disk. Do not modify counts while accumulation is active.
    """
    def __init__(self, bounds, shape=(2048, 2048), *, counts=None):
        self.bounds = _bounds(bounds)
        self.shape = _shape(shape)
        if counts is None:
            counts = np.zeros(self.shape, dtype=np.uint64)
        if counts.shape != self.shape or counts.dtype != np.uint64 or not counts.flags.writeable:
            raise ValueError("counts must be a writable uint64 array of the requested shape")
        if np.any(counts):
            raise ValueError("provide a zeroed count array for a new histogram")
        self.counts = counts
        self.seen = self.accepted = self.outside = self.nonfinite = 0

    def add(self, xy):
        xy = _points(xy)
        # With unit weights, no individual bin can exceed this total.
        if self.seen + len(xy) > np.iinfo(np.uint64).max:
            raise OverflowError("histogram draw count exceeds uint64")
        finite = np.isfinite(xy).all(axis=1)
        inside = finite & (xy >= self.bounds[0]).all(axis=1) & (xy <= self.bounds[1]).all(axis=1)
        selected = xy[inside]
        norm = (selected - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        height, width = self.shape
        cols = np.clip((norm[:, 0] * width).astype(np.int64), 0, width - 1)
        rows = np.clip(((1 - norm[:, 1]) * height).astype(np.int64), 0, height - 1)
        # Unique within a batch avoids one dense grid-sized bincount per batch.
        ids, increments = np.unique(rows * width + cols, return_counts=True)
        self.counts[ids // width, ids % width] += increments.astype(np.uint64)
        self.seen += len(xy)
        self.accepted += len(selected)
        self.nonfinite += int((~finite).sum())
        self.outside += int((finite & ~inside).sum())

    def receipt(self):
        return dict(shape=list(self.shape), bounds=self.bounds.tolist(), seen=self.seen,
                    accepted=self.accepted, outside=self.outside, nonfinite=self.nonfinite,
                    count_dtype="uint64", orientation="row 0 = maximum Y; columns increase with X")


@dataclass
class CoarseCounts:
    counts: np.ndarray
    # Actual original-bin area, including partial right/bottom edge cells.
    areas: np.ndarray


def coarsen_counts(counts, factor=2):
    """Sum integer counts before coloring; never downsample an RGB image.

    factor is the reduction along *each* axis. Partial edge blocks retain their
    actual area. To coarsen again, use the original counts with the total factor.
    """
    factor = operator.index(factor)
    counts = np.asarray(counts)
    if factor < 1 or counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("expected a nonempty 2D count array and positive factor")
    if counts.dtype.kind not in "ui" or np.any(counts < 0):
        raise ValueError("counts must be nonnegative integers")
    # Sum row totals as Python ints to detect overflow even on uint64 inputs.
    limit = np.iinfo(np.uint64).max
    if int(counts.max()) * counts.size > limit:
        if sum(map(int, counts.flat)) > limit:
            raise OverflowError("coarsened total exceeds uint64")
    ys = np.arange(0, counts.shape[0], factor)
    xs = np.arange(0, counts.shape[1], factor)
    reduced = np.add.reduceat(np.add.reduceat(counts, ys, axis=0, dtype=np.uint64), xs, axis=1, dtype=np.uint64)
    areas = np.minimum(factor, counts.shape[0] - ys)[:, None] * np.minimum(factor, counts.shape[1] - xs)[None, :]
    return CoarseCounts(reduced, areas)
