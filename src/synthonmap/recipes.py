"""Uniform draws from a union of reaction-role Cartesian products."""
import math

import numpy as np


class RecipeSpace:
    """Each route contains 1..4 ordered pools of positive synthon indices.

    Index zero means padding. Route probabilities are proportional to their
    number of combinations, not uniform over routes. IDs identify recipes,
    not unique or necessarily chemically valid assembled molecules.
    """
    def __init__(self, routes):
        self.routes = [[np.asarray(pool) for pool in route] for route in routes]
        if not self.routes or any(not 1 <= len(route) <= 4 for route in self.routes):
            raise ValueError("provide routes with one to four pools")
        for route in self.routes:
            for pool in route:
                if pool.ndim != 1 or not len(pool) or pool.dtype.kind not in "iu" or np.any(pool <= 0):
                    raise ValueError("pools must contain positive synthon indices")
                if int(pool.max()) > np.iinfo(np.int64).max:
                    raise OverflowError("synthon indices exceed int64")
                if len(np.unique(pool)) != len(pool):
                    raise ValueError("a role pool contains duplicate indices")
        self.routes = [[pool.astype(np.int64) for pool in route] for route in self.routes]
        capacities = [math.prod(map(len, route)) for route in self.routes]
        self.total = sum(capacities)
        if self.total > np.iinfo(np.int64).max:
            raise OverflowError("recipe IDs require a space of at most 2**63-1 combinations")
        self.ends = np.cumsum(capacities, dtype=np.int64)
        self.starts = self.ends - capacities

    def decode(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.total):
            raise ValueError("invalid recipe IDs")
        ids = ids.astype(np.int64)
        reactions = np.searchsorted(self.ends, ids, side="right")
        slots = np.zeros((len(ids), 4), dtype=np.int64)
        for reaction, pools in enumerate(self.routes):
            selected = np.flatnonzero(reactions == reaction)
            remainder = ids[selected] - self.starts[reaction]
            for slot, pool in enumerate(pools):
                slots[selected, slot] = pool[remainder % len(pool)]
                remainder //= len(pool)
        return reactions, slots

    def sample(self, rng, count):
        ids = rng.integers(self.total, size=count, dtype=np.int64)
        reactions, slots = self.decode(ids)
        return ids, reactions, slots
