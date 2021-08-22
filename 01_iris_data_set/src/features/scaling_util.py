from sklearn.preprocessing import StandardScaler


def scale_feature(X):
    """The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean
    value 0 and standard deviation of 1.

    @see https://stackoverflow.com/a/40767144

    TODO: X
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    return scaled_data
