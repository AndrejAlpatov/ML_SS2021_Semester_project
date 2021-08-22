import pytest
import numpy as np

from fitting.point_line_distance import get_distances


@pytest.yield_fixture
def u_v_points_distances():
    u = np.array([1., 1., 0.])
    v = np.array([0., 0., 1.])
    points = np.array([[1., 2., -1.],
                       [1., 2., 0.],
                       [1., 2., 1.],
                       [1., 2., 2.]])
    distances = np.array([1., 1., 1., 1.])
    yield u, v, points, distances
    del u
    del v
    del points
    del distances


class TestGetDistances(object):
    def test_get_distances(self, u_v_points_distances):
        u, v, points, distances_expected = u_v_points_distances
        distances_actual = get_distances(u, v, points)
        assert np.allclose(distances_actual, distances_expected)
