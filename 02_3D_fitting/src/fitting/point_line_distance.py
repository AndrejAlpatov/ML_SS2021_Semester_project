import numpy as np


def get_distances(u, v, points):
    """
    I am trying to get the distance from P3 perpendicular
    to a line drawn between P1 and P2.

    d=np.cross(p2-p1,p3-p1)/norm(p2-p1)
    p1 = u
    p2 = u + v
    p3 = p3
    p2 - p1 = u + v - u = v

    See also: https://stackoverflow.com/q/39840030

    :param u: Position vector of line.
    :param v: Line parallel to this vector.
    :param points: Array of points.
    :return: Array of distances from the by 'u + t * v'
    defined line to the array of 'points'.
    """
    distance_vectors = np.cross(v, points - u) / np.linalg.norm(v)
    return np.linalg.norm(distance_vectors, axis=1)
