import numpy as np
import pandas as pd
import sys
sys.path.append('../../src/features/')
import add_column_with_count_of_elements


class TestAddColumnWithCountOfElements(object):

    DF_TEST = pd.DataFrame({'name': ['Earth', 'Moon', 'Mars', 'Mars', 'Moon'],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606]})
    COLUMNS = ['name', 'mass_to_earth']

    def test_add_column_with_count_of_elements(self, df = DF_TEST):

        # Arrange

        expected = (3, np.array([1,2,2,2,2]).all())

        # Act
        df_temp = add_column_with_count_of_elements.add_column_with_count_of_elements(df, 'name', 'new_column_name')
        actual = (len(df_temp.columns), df_temp['new_column_name'].values.all())

        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message





