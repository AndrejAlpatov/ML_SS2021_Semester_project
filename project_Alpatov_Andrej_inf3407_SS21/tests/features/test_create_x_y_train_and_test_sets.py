import pandas as pd
import sys
sys.path.append('../../src/features/')
import create_x_y_train_and_test_sets


class TestCreateXYTrainAndTestSets(object):

    DF_TEST = pd.DataFrame({'name': ['Earth', 'Moon', 'Mars', 'Mars', 'Moon', 'Moon'],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606, 0.606],
                            'extra': [5, 8, 7, 3, 4, 2]})

    def test_create_x_y_train_and_test_sets(self, df = DF_TEST, target = 'mass_to_earth', predict=['extra'],
                                            test_size=0.5, random_state=42):

        # Arrange
        expected = (True, 3, 1, 1, 1, 1)

        # Act
        y, x, x_train, x_test, y_train, y_test = \
            create_x_y_train_and_test_sets.create_x_y_train_and_test_sets(df, target, predict, test_size, random_state)

        actual = ((y == [1, 0.606, 0.107, 0.107, 0.606, 0.606]).all(), 3, 1, 1, 1, 1)
        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message





