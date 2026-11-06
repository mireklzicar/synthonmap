"""Log-density coloring. Raw counts are independent of display saturation."""
from pathlib import Path

import numpy as np


def _density(counts, areas):
    counts = np.asarray(counts)
    if counts.ndim != 2 or not counts.size or counts.dtype.kind not in "ui" or np.any(counts < 0):
        raise ValueError("counts must be a nonempty 2D array of nonnegative integers")
    areas = np.asarray(areas, dtype=np.float64)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("areas must be positive and finite")
    if areas.ndim and areas.shape != counts.shape:
        raise ValueError("areas must be scalar or match counts")
    return counts.astype(np.float64) / areas


def display_ceiling(counts, *, quantile=0.995, areas=1, max_side=2048):
    """Positive-bin quantile on a spatially strided sample; display only.

    Fix and reuse the returned ceiling when comparing coarsening levels. This
    does not change saved counts. The stride is recorded by the caller if needed.
    """
    if not 0 < quantile <= 1 or max_side < 1:
        raise ValueError("invalid display quantile/sample size")
    counts = np.asarray(counts)
    if counts.ndim != 2 or not counts.size:
        raise ValueError("expected nonempty 2D counts")
    stride = max(1, int(np.ceil(max(counts.shape) / max_side)))
    sample = counts[::stride, ::stride]
    sampled_areas = np.asarray(areas)[::stride, ::stride] if np.ndim(areas) else areas
    values = _density(sample, sampled_areas)
    positive = values[values > 0]
    return max(2.0, float(np.quantile(positive, quantile))) if len(positive) else 2.0


def colorize(counts, *, ceiling, areas=1, palette="viridis"):
    """RGB uint8: LUT[round(255 * clip(log1p(count/area)/log1p(ceiling)))]."""
    from matplotlib import colormaps

    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("ceiling must be finite and positive")
    values = np.log1p(_density(counts, areas)) / np.log1p(ceiling)
    lut = np.round(colormaps[palette](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut[np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)]


def save_png(path, counts, *, ceiling, areas=1, palette="viridis", rows_per_chunk=256):
    """Paint in strips to bound floating temporaries; PIL retains the RGB image.

    Output is an ordinary PNG, without tiles or a viewer. For 32768² grids,
    budget for uint64 counts (8 GiB) and RGB/image storage in addition.
    """
    from PIL import Image

    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    counts = np.asarray(counts)
    if counts.ndim != 2 or not counts.size:
        raise ValueError("expected nonempty 2D counts")
    area_array = np.asarray(areas)
    if area_array.ndim and area_array.shape != counts.shape:
        raise ValueError("areas must be scalar or match counts")
    height, width = counts.shape
    image = Image.new("RGB", (width, height))
    for y in range(0, height, rows_per_chunk):
        area = area_array[y:y + rows_per_chunk] if area_array.ndim else areas
        rgb = colorize(counts[y:y + rows_per_chunk], ceiling=ceiling, areas=area, palette=palette)
        image.paste(Image.fromarray(rgb), (0, y))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
