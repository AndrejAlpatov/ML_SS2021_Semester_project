import pandas as pd
import numpy as np
import sys
sys.path.append('../../src/features/')
import grouping_dataframe

class TestGroupingDataframe(object):

    DF_TEST = pd.DataFrame({'name': ['Earth', 'Moon', 'Mars', 'Mars', 'Moon'],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606]})
    COLUMNS = ['name', 'mass_to_earth']
    FUNCTIONS = [np.mean, np.std]

    # The check matches the number of columns and the presence of the NaN values in return value
    def test_grouping_dataframe_(self, df=DF_TEST, columns=COLUMNS, functions=FUNCTIONS):

        # Arrange
        columns_for_grouping = [columns[0]]
        columns_for_functions_to_be_applied = [columns[1]]
        df_temp = grouping_dataframe.grouping_dataframe_(df, columns_for_grouping, columns_for_functions_to_be_applied,
                                                              functions)
        columns_number = len(df.columns)-len(columns_for_functions_to_be_applied) + \
                        len(columns_for_functions_to_be_applied)*len(functions)

            # Expected no NaN values and column_number number of columns
        expected = (False, columns_number)

        # Act
        actual = (df.isnull().values.any(), len(df_temp.columns))
        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message







