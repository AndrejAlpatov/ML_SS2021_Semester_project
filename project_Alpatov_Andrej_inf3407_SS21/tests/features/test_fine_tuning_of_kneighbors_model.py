import numpy as np
import pandas as pd
import sys
sys.path.append('../../src/features/')
import fine_tuning_of_kneighbors_model as ftm


class TestFineTuningOfKneighborsModel(object):

    DF_TEST = pd.DataFrame({'name': ['Earth', 'Moon', 'Mars', 'Mars', 'Moon', 'Moon', 'Moon', 'Moon', 'Earth', 'Earth'],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606, 0.606, 0.606, 0.606, 1, 1],
                            'extra': [5, 8, 7, 3, 4, 2, 8, 4, 9, 0]})

    def test_fine_tuning_of_kneighbors_model(self, x_train=DF_TEST[['extra', 'mass_to_earth']], y_train=DF_TEST['name'],
                               x_test=DF_TEST[['extra', 'mass_to_earth']], y_test=DF_TEST['name']):

        # Arrange

        expected = (type(np.array([])), type(np.array([])), type(np.array([])))

        # Act
        neighbors, train_accuracy, test_accuracy = ftm.fine_tuning_of_kneighbors_model(x_train, y_train, x_test, y_test)
        actual = (type(neighbors), type(train_accuracy), type(test_accuracy))

        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message



