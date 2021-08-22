import numpy as np
import math

DELTA = 1e-10


def enclosing_angle(u, v):
    """Calculate the enclosing angle by utilising the 'scalar product'.

    :param u: First vector.
    :param v: Second vector.
    :return: The enclosing angle in radian between vector 'u' and 'v'.
    """
    global DELTA
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm < DELTA or v_norm < DELTA:
        raise ZeroDivisionError()
    return math.acos(np.dot(u, v) / (u_norm * v_norm))


def standard_parametric_vector(a, p):
    """Returns position vector 'u' with corresponding
    direction vector 'v' where 'u' is in the x, y plane
    (z == 0). Works only for 3d vectors.

    Calculated by cartesian equations for the straight
    line through 'a' with direction rations 'p'.

    :param a: The straight line must go through 'a'.
    :param p: The straight line has 'p/|p|' direction ratios.
    :return: Position vector 'u' with corresponding unit
    direction vector 'v'.
    """
    global DELTA
    if np.abs(p[2]) < DELTA:
        raise ZeroDivisionError('Position vector may not hold a zero z component!'
                                ' This line will never cross the x, y plane.')
    u = np.zeros(3)
    u[0] = a[0] - a[2] * p[0] / p[2]
    u[1] = a[1] - a[2] * p[1] / p[2]
    # TODO raise an exception if 'p' is a null vector
    p_unit = p / np.linalg.norm(p)
    return u, p_unit
