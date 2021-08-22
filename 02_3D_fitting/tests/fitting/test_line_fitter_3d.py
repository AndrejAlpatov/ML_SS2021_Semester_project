import pytest
import numpy as np
import pandas as pd

from fitting.line_fitter_3d import fit_line, get_line
from visualization.cloud_visualization import plot_3d_line
from fitting.vector_geometry_helper import standard_parametric_vector


# noinspection PyUnresolvedReferences

# @pytest.fixture
# def get_point_cloud():
#     x = np.mgrid[-2:5:120j]
#     y = np.mgrid[1:9:120j]
#     z = np.mgrid[-5:3:120j]
#     data = np.concatenate((x[:, np.newaxis],
#                            y[:, np.newaxis],
#                            z[:, np.newaxis]),
#                           axis=1)
#     yield data
#     del data


@pytest.yield_fixture
def point_cloud_with_noise():
    u = np.array([0., 0., 0.])[:, np.newaxis]
    v = np.array([1., 1., 1.])[:, np.newaxis]
    points = get_point_cloud_helper(u, v, gaussian_noise_sigma=0.4)
    yield points, u, v
    del points


@pytest.yield_fixture
def point_cloud():
    u = np.array([0., 0., 0.])[:, np.newaxis]
    v = np.array([1., 1., 1.])[:, np.newaxis]
    points = get_point_cloud_helper(u, v, gaussian_noise_sigma=0.)
    yield points, u, v
    del points


@pytest.yield_fixture
def point_cloud_dataframe():
    u = np.array([0., 0., 0.])[:, np.newaxis]
    v = np.array([1., 1., 1.])[:, np.newaxis]
    points = get_point_cloud_helper(u, v, gaussian_noise_sigma=0.)
    df_points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    yield df_points, u, v
    del points


def get_point_cloud_helper(u, v, length=10., number_of_points=100, gaussian_noise_sigma=0.4):
    points = u + v * length * np.linspace(0., length, number_of_points)
    points += np.random.normal(size=points.shape) * gaussian_noise_sigma
    return points.T


DELTA = 1e-5


class TestFitLine(object):
    def test_fit_line(self, point_cloud_with_noise):
        points, u_exp, v_exp = point_cloud_with_noise
        u, v = fit_line(points)
        line_points = get_line(points, u, v)
        plot_3d_line(points, line_points)

    def test_fit_line_0(self, point_cloud_dataframe):
        points, u_expected, v_expected = point_cloud_dataframe
        u_expected_standard, v_expected_standard = standard_parametric_vector(u_expected, v_expected)
        u_actual, v_actual = fit_line(points)
        u_actual_standard, v_actual_standard_unit = standard_parametric_vector(u_actual, v_actual)
        assert pytest.approx(u_expected_standard) == u_actual_standard
        assert pytest.approx(v_expected_standard, rel=DELTA) == v_actual_standard_unit[:, np.newaxis]

    def test_fit_line_1(self, point_cloud):
        global DELTA
        points, u_expected, v_expected = point_cloud
        u_expected_standard, v_expected_standard = standard_parametric_vector(u_expected, v_expected)
        u_actual, v_actual = fit_line(points)
        u_actual_standard, v_actual_standard_unit = standard_parametric_vector(u_actual, v_actual)
        assert pytest.approx(u_expected_standard) == u_actual_standard
        assert pytest.approx(v_expected_standard, rel=DELTA) == v_actual_standard_unit[:, np.newaxis]

    def test_fit_line_2(self, point_cloud_with_noise):
        """Note the absolute error 'abs=0.5' which is needed with gaussian random noise.

        :param point_cloud_with_noise: fixture
        :return: void
        """
        points, u_expected, v_expected = point_cloud_with_noise
        u_expected_standard, v_expected_standard = standard_parametric_vector(u_expected, v_expected)
        u_actual, v_actual = fit_line(points)
        u_actual_standard, v_actual_standard_unit = standard_parametric_vector(u_actual, v_actual)
        assert pytest.approx(u_expected_standard, abs=0.5) == u_actual_standard
        assert pytest.approx(v_expected_standard, abs=0.5) == v_actual_standard_unit[:, np.newaxis]
