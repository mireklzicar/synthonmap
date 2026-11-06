import numpy as np
from PIL import Image
import pytest

from synthonmap import Histogram, fit_bounds, coarsen_counts, colorize, save_png


def test_stream_matches_numpy_and_accounts_for_every_draw():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-2, 2, (10007, 2))
    xy = np.r_[xy, [[-1, -1], [1, 1], [0, 0], [-1, 1], [1, -1], [np.nan, 0], [np.inf, 0]]]
    hist = Histogram([[-1, -1], [1, 1]], (13, 17))
    for batch in np.array_split(xy, 23):
        hist.add(batch)
    finite = np.isfinite(xy).all(1)
    inside = finite & (np.abs(xy) <= 1).all(1)
    # Independent NumPy reference: histogram the inverted Y directly, including
    # its convention for exact inner edges and the outermost right endpoint.
    expected = np.histogram2d(-xy[inside, 1], xy[inside, 0], bins=(13, 17), range=[[-1, 1], [-1, 1]])[0]
    np.testing.assert_array_equal(hist.counts, expected)
    assert hist.seen == hist.accepted + hist.outside + hist.nonfinite == len(xy)
    assert hist.accepted == int(hist.counts.sum()) == int(inside.sum())
    assert hist.nonfinite == 2


def test_many_coincident_points_and_memmap(tmp_path):
    counts = np.lib.format.open_memmap(tmp_path / "counts.npy", mode="w+", dtype=np.uint64, shape=(2, 2))
    hist = Histogram([[0, 0], [1, 1]], (2, 2), counts=counts)
    hist.add(np.tile([.1, .9], (100000, 1)))
    hist.add(np.tile([.1, .9], (100000, 1)))
    counts.flush()
    assert np.load(tmp_path / "counts.npy")[0, 0] == 200000


def test_bounds_replay_includes_tails_and_constant_axis():
    points = np.array([[2, -5], [2, 3], [2, 10000], [np.nan, 0]])
    bounds = fit_bounds([points[:2], points[2:]])
    hist = Histogram(bounds, (7, 11)); hist.add(points)
    assert hist.accepted == 3 and hist.nonfinite == 1 and hist.outside == 0
    assert bounds[1, 0] - bounds[0, 0] == pytest.approx(bounds[1, 1] - bounds[0, 1])
    with pytest.raises(ValueError):
        fit_bounds([np.empty((0, 2))])


def test_reduction_preserves_mass_and_partial_edge_area():
    counts = np.arange(35, dtype=np.uint64).reshape(5, 7)
    reduced = coarsen_counts(counts, 3)
    expected = np.array([[counts[y:y+3, x:x+3].sum() for x in range(0, 7, 3)] for y in range(0, 5, 3)])
    np.testing.assert_array_equal(reduced.counts, expected)
    np.testing.assert_array_equal(reduced.areas, [[9, 9, 3], [6, 6, 2]])
    assert int(reduced.counts.sum()) == int(counts.sum())
    huge = np.array([[2**32, 2**32]], dtype=np.uint64)
    assert coarsen_counts(huge, 2).counts[0, 0] == 2**33
    with pytest.raises(OverflowError):
        coarsen_counts(np.array([[2**64-1, 1]], dtype=np.uint64), 2)


def test_coloring_retains_counts_and_uses_mean_density(tmp_path):
    counts = np.full((5, 7), 3, dtype=np.uint64)
    before = counts.copy()
    coarse = coarsen_counts(counts, 3)
    expected = colorize(counts, ceiling=5)[0, 0]
    actual = colorize(coarse.counts, ceiling=5, areas=coarse.areas)
    np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
    save_png(tmp_path / "density.png", counts, ceiling=5, rows_per_chunk=2)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "density.png")), colorize(counts, ceiling=5))
    np.testing.assert_array_equal(counts, before)


@pytest.mark.parametrize("bounds", [[[0, 0], [0, 1]], [[0, 0], [np.inf, 1]], [0, 1]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        Histogram(bounds)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_matches_numpy(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from synthonmap.torch_histogram import TorchHistogram
    rng = np.random.default_rng(19)
    xy = np.r_[rng.normal(size=(10001, 2)), [[-1, -1], [1, 1], [0, 0], [np.nan, 0], [np.inf, 1]]]
    cpu = Histogram([[-1, -1], [1, 1]], (32, 64))
    other = TorchHistogram(cpu.bounds, cpu.shape, device=device)
    for batch in np.array_split(xy, 13):
        cpu.add(batch); other.add(torch.as_tensor(batch, device=device))
    np.testing.assert_array_equal(cpu.counts, other.numpy())
    for key in ["seen", "accepted", "outside", "nonfinite"]:
        assert cpu.receipt()[key] == other.receipt()[key]
