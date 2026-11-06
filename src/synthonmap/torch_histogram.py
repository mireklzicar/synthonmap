"""Optional batched CPU/CUDA accumulator; importing synthonmap needs no torch."""
import numpy as np
import torch

from .histogram import _bounds, _shape


class TorchHistogram:
    """int64 scatter-add on the selected device, with one final host transfer.

    Binning uses float64 on CPU and CUDA for agreement with Histogram. Inference
    can still be float32. Counts are capped at 2**63-1 draws; they never silently
    wrap as an int32 histogram would above roughly 2.1 billion coincident draws.
    """
    def __init__(self, bounds, shape=(2048, 2048), *, device="cuda"):
        self.bounds = _bounds(bounds)
        self.shape = _shape(shape)
        self.device = torch.device(device)
        self.limits = torch.as_tensor(self.bounds, device=self.device)
        self.counts = torch.zeros(self.shape, dtype=torch.int64, device=self.device)
        self.stats = torch.zeros(3, dtype=torch.int64, device=self.device)
        self.seen = 0

    @torch.no_grad()
    def add(self, xy):
        xy = torch.as_tensor(xy, dtype=torch.float64, device=self.device)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("coordinates must have shape (N, 2)")
        if self.seen + len(xy) > np.iinfo(np.int64).max:
            raise OverflowError("histogram draw count exceeds int64")
        finite = torch.isfinite(xy).all(1)
        inside = finite & (xy >= self.limits[0]).all(1) & (xy <= self.limits[1]).all(1)
        norm = (xy[inside] - self.limits[0]) / (self.limits[1] - self.limits[0])
        height, width = self.shape
        cols = (norm[:, 0] * width).long().clamp(0, width - 1)
        rows = ((1 - norm[:, 1]) * height).long().clamp(0, height - 1)
        ids = rows * width + cols
        self.counts.view(-1).scatter_add_(0, ids, torch.ones_like(ids))
        self.stats += torch.stack([inside.sum(), (finite & ~inside).sum(), (~finite).sum()])
        self.seen += len(xy)

    def numpy(self):
        return self.counts.cpu().numpy().astype(np.uint64)

    def receipt(self):
        accepted, outside, nonfinite = self.stats.cpu().tolist()
        return dict(shape=list(self.shape), bounds=self.bounds.tolist(), seen=self.seen,
                    accepted=accepted, outside=outside, nonfinite=nonfinite,
                    count_dtype="int64", device=str(self.device),
                    orientation="row 0 = maximum Y; columns increase with X")
