import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as m3d

FIG_SIZE = (12, 12)


def plot_3d_line(points, line_points):
    global FIG_SIZE
    ax = m3d.Axes3D(plt.figure(figsize=FIG_SIZE))
    ax.scatter3D(*points.T)
    ax.plot3D(*line_points.T)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_3d_cloud(points, title=''):
    global FIG_SIZE
    fig = plt.figure(figsize=FIG_SIZE)
    ax = Axes3D(fig)
    ax.scatter(points['posX'], points['posY'], points['posZ'])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()
