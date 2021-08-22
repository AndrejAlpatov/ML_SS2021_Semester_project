import numpy as np
import pytest

from fitting.vector_geometry_helper import enclosing_angle, standard_parametric_vector


class TestEnclosingAngle(object):
    def test_enclosing_angle(self):
        u = np.array([0., 0., 1.])
        v = np.array([0., 1., 0.])
        assert enclosing_angle(u, v) == pytest.approx(np.pi / 2.)

    def test_enclosing_angle_1(self):
        u = np.array([0., 1., 0.])
        v = np.array([0., 0., 1.])
        assert enclosing_angle(u, v) == pytest.approx(np.pi / 2.)

    def test_enclosing_angle_2(self):
        u = np.array([0., 1., 0.])
        v = np.array([0., 1., 0.])
        assert enclosing_angle(u, v) == pytest.approx(0.)

    def test_enclosing_angle_3(self):
        u = np.array([0., -1., 0.])
        v = np.array([0., 1., 0.])
        assert enclosing_angle(u, v) == pytest.approx(np.pi)

    def test_zero_division(self):
        u = np.array([0., 0., 0.])
        v = np.array([0., 1., 0.])
        with pytest.raises(ZeroDivisionError):
            enclosing_angle(u, v)
        u = np.array([0., 1., 0.])
        v = np.array([0., 0., 0.])
        with pytest.raises(ZeroDivisionError):
            enclosing_angle(u, v)
        u = np.array([0., 0., 0.])
        v = np.array([0., 0., 0.])
        with pytest.raises(ZeroDivisionError):
            enclosing_angle(u, v)


class TestStandardParametricVector(object):
    def test_standard_parametric_vector(self):
        a = np.array([0., 0., 0.])
        p = np.array([1., 1., 1.])
        u, v_unit = standard_parametric_vector(a, p)
        assert u == pytest.approx(a)
        assert v_unit == pytest.approx(p/np.linalg.norm(p))

    def test_standard_parametric_vector_1(self):
        a = np.array([1., 1., 0.])
        p = np.array([1., 1., 1.])
        u, v_unit = standard_parametric_vector(a, p)
        assert u == pytest.approx(a)
        assert v_unit == pytest.approx(p/np.linalg.norm(p))

    def test_standard_parametric_vector_2(self):
        a = np.array([1., 1., 1.])
        u_expected = np.array([0., 0., 0.])
        p = np.array([1., 1., 1.])
        u, v_unit = standard_parametric_vector(a, p)
        assert u == pytest.approx(u_expected)
        assert v_unit == pytest.approx(p/np.linalg.norm(p))
