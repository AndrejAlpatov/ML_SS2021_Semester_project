import models.k_neighbors_model as knm
import numpy as np


# Makes a K-Neighbors model with scaling
def fine_tuning_of_kneighbors_model(X_train, y_train, X_test, y_test):

    """

    :param X_train: np-array with predictor-values for train set
    :param y_train: np-array with target-values for train set
    :param X_test: np-array with predictor-values for test set
    :param y_test: np-array with target-values for test set
    :return: three np-arrays with neighbors(used for parameter k_neighbors), train and test accuracy
    """

    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):

        # Fit the pipeline to the training set
        knn_scaled = knm.k_neighbors_model(k, X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn_scaled.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn_scaled.score(X_test, y_test)

    return neighbors, train_accuracy, test_accuracy

