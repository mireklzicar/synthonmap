# From coordinates to density

## Accumulate before coloring

For Cartesian bounds `lo=(xmin,ymin)`, `hi=(xmax,ymax)` and image shape `(H,W)`:

```text
inside = finite(x,y) and lo <= (x,y) <= hi
u = ((x,y) - lo) / (hi - lo)
column = min(floor(u.x * W), W - 1)
row    = min(floor((1 - u.y) * H), H - 1)
counts[row, column] += 1
```

Row zero is the top of the image (largest Y); there is no later vertical flip.
Exact outer boundary points are included. Internal Y boundaries follow the image
coordinate floor convention above, which differs from first histogramming upward
Y and then flipping the result at exact boundaries. Finite points outside the bounds
are counted as outside and dropped; they are never pushed into edge bins. Nonfinite
rows have a separate counter. `seen = accepted + outside + nonfinite`, and the grid
sum equals accepted draws.

The CPU implementation groups repeated pixel indices within each batch, then adds
their integer multiplicities. It does not allocate a dense grid per batch. CUDA
uses `scatter_add_` into a persistent int64 tensor. Both bin in float64, including
float32 model predictions converted to float64, to keep CPU/CUDA binning consistent.
This is a deliberate improvement over the original renderer's float32 GPU binning;
points extremely near boundaries may differ from legacy artifacts.

## Bounds and replay

`fit_bounds` makes one pass over finite coordinates and fits a padded square around
their extrema. A constant axis receives a nonzero span. The coordinate generator
must be replayable: reset the seed and keep the model, inference precision, device,
batch size and parameters fixed. Then a second pass accumulates those same draws.
This doubles inference work but needs no full coordinate file. A known fixed domain
can skip the first pass; its outside/nonfinite counters make lost mass explicit.

The worked example has no quantile cropping. If a caller chooses cropped bounds,
the receipt records the excluded points. Cropping changes what the histogram covers.

## Coarsen the integer field

For a factor `f`, sum each `f × f` block. Never resize the colored image to obtain
a scientific lower-resolution count map. `coarsen_counts` also returns the original
bin area of every block, including partial right/bottom edges for odd dimensions.
The sum of all coarse counts equals the original sum. Factor 8 means eight times
fewer rows and columns (64 times fewer cells); factor 64 means 4,096 times fewer cells.
The last partial blocks have less coordinate width/height; regular PNG pixels cannot
express that irregular geometry exactly. Use divisible dimensions for comparing PNGs,
or retain original bounds, dimensions and factors for quantitative analysis.

Color uses **mean count per original bin** at each coarsening level:

```text
density = count / original_bin_area
value = clip(log1p(density) / log1p(display_ceiling), 0, 1)
RGB = viridis_LUT[round(255 * value)]
```

The ceiling defaults to the 99.5th percentile of positive-bin densities in a
spatially strided sample of at most 2048 bins per axis, floored at 2. This affects
display saturation only. Keep one ceiling when comparing resolutions of one grid;
do not fit a new ceiling for every image. Across independently fitted maps, bin
sizes and coordinate distortions differ, so equal colors do not imply equal
chemical-space density.

`counts.npy` is the measurement. PNG is a lossy display representation with log
transfer and saturation; its pixel values cannot recover the original counts.

## Memory and large runs

Persistent memory is O(H×W + batch size), not O(number of draws). A 2048² uint64/int64
histogram takes 32 MiB; 8192² takes 512 MiB; 32768² takes 8 GiB. All counts use unit
weights. CPU accumulation rejects more than 2**64−1 total draws; PyTorch rejects
more than 2**63−1. These total bounds also protect each bin from overflow. This
avoids an int32 bin wrapping if billions of points happen to coincide.

CPU accumulation can use a fresh, zeroed `.npy` memory map:

```python
counts = np.lib.format.open_memmap(
    "counts.npy", mode="w+", dtype=np.uint64, shape=(8192, 8192))
hist = Histogram(bounds, counts.shape, counts=counts)
for xy in batches:
    hist.add(xy)
counts.flush()
```

GPU accumulation avoids transferring every coordinate batch to the host:

```python
from synthonmap.torch_histogram import TorchHistogram

hist = TorchHistogram(bounds, (8192, 8192), device="cuda")
for xy in predicted_cuda_batches:
    hist.add(xy)
counts = hist.numpy()  # one final transfer; caller needs host memory too
```

`save_png` colors strips to avoid a full floating-point image, but PIL still holds
the output image in memory (approximately 3–4 bytes/pixel, plus encoding buffers).
Coarsening allocates intermediate/output arrays. For very large rasters, save raw
counts and export a coarsened PNG, or budget for these additional allocations.
The library does not reserve a GPU fraction, kill other jobs, or automatically
launch a billion-point run.

The example's NumPy recipe sampler and CPU UMAP favor portability and readability;
they are not the fully GPU-resident production sampler. Scaling training to 250K
products requires an appropriate neighbor-search/UMAP backend and memory budget;
changing `--train` alone is not a GPU training implementation.
