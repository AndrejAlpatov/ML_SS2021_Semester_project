import pandas as pd
import sys
sys.path.append('../../src/models/')
import k_neighbors_model


class TestKNeighborsModel(object):

    DF_TEST = pd.DataFrame({'name': [1, 1, 1, 1, 1, 1],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606, 0.606],
                            'extra': [5, 8, 7, 3, 4, 2]})
    COLUMNS = ['name', 'mass_to_earth']

    def test_k_neighbors_model(self, n_neighbors_in=3, x_train=DF_TEST[['extra', 'mass_to_earth']],
                               y_train=DF_TEST['name']):

        # Arrange
        expected = 1

        # Act
        temp = k_neighbors_model.k_neighbors_model(n_neighbors_in, x_train, y_train)
        actual = temp.score(x_train,y_train)
        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message





