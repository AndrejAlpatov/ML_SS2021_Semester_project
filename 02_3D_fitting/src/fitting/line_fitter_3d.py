import numpy as np


def fit_line(data):
    """Fit a 3D line in a 3D point cloud.

    :param data: 3D point cloud data. A 3D numpy array with
                rows filled with x, y and y values. The shape
                must be (3, n).
    :return: two 3D vectors 'u' (position) and 'v' (direction)
                describing the 3D line fit
    """
    data_mean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - data_mean)
    return data_mean, vv[0]


def get_line(data, u, v):
    """Calculate beginning and end of a straight
    line with position vector 'u' and parallel to
    vector 'v' determined by the 'data' point cloud
    in z direction.

    :param data: Point cloud.
    :param u: Position vector.
    :param v: Direction vector.
    :return: Beginning and end coordinates of a straight
    line with position vector 'u' and parallel to
    vector 'v' determined by the 'data' point cloud
    in z direction.
    """
    z_min = data[:, 2].min()
    z_max = data[:, 2].max()
    line_points = v * np.mgrid[z_min:z_max:2j][:, np.newaxis]
    line_points += u
    return line_points
