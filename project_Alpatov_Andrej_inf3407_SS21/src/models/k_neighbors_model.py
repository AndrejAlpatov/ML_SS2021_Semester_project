from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Makes a K-Neighbors model with scaling
def k_neighbors_model(n_neighbors_in, x_train_in, y_train_in):

    """

    :param n_neighbors_in: Int Parameter n_neighbors for KNeighborsClassifier
    :param x_train_in: train set with predictor variables
    :param y_train_in: train set with target variables
    :return: sklearn.pipeline.Pipeline with KNeighborsClassifier
    """

    # Setup the pipeline steps: steps
    steps = [('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier(n_neighbors=n_neighbors_in))]

    # Create the pipeline: pipeline
    pipeline = Pipeline(steps)

    # Fit the pipeline to the training set
    knn_scaled = pipeline.fit(x_train_in, y_train_in)

    return knn_scaled

