from itertools import product

import numpy as np
import pytest

from synthonmap.recipes import RecipeSpace


def test_decode_is_bijection_and_sampling_weights_routes_by_capacity():
    routes = [[[1, 2], [3, 4, 5]], [[6], [7], [8, 9]]]
    space = RecipeSpace(routes)
    assert space.total == 8
    rs, slots = space.decode(np.arange(8))
    actual = {(r, tuple(row[row != 0])) for r, row in zip(rs, slots)}
    expected = {(r, combo) for r, pools in enumerate(routes) for combo in product(*pools)}
    assert actual == expected
    ids, rs, slots = space.sample(np.random.default_rng(5), 80000)
    assert abs((rs == 0).mean() - .75) < .01
    np.testing.assert_array_equal(space.decode(ids)[1], slots)


def test_invalid_space_and_ids():
    with pytest.raises(ValueError):
        RecipeSpace([[[1.5]]])
    with pytest.raises(ValueError):
        RecipeSpace([[[0]]])
    with pytest.raises(ValueError):
        RecipeSpace([[[1, 1]]])
    with pytest.raises(ValueError):
        RecipeSpace([[[1]]]).decode(np.array([1]))
