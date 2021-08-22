from matplotlib import pyplot as plt


# Generation of a plot. It is possible to plot an unlimited number of curves on one graph.
def plot_generator(labels, lines):

    """

    :param labels: List, consists of three items: title, x-axis label, y-axis label
    :param lines: List of tuples. Each tuple consists of three elements: Coordinates for the x-axis, for the y-axis,
    and the name of the curve
    :return: void
    """

    # Generate plot
    plt.title(labels[0])
    for x in lines:
        plt.plot(x[0], x[1], label = x[2])

    plt.legend()
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.show()



