"""Counts first, colors second. No web server or tiling dependency."""
from .histogram import Histogram, fit_bounds, coarsen_counts
from .render import display_ceiling, colorize, save_png

__all__ = ["Histogram", "fit_bounds", "coarsen_counts", "display_ceiling", "colorize", "save_png"]
