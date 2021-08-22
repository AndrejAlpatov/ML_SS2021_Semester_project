import sys
import pandas as pd

sys.path.append('../../src/features/')
import dataframe_info


class TestDataframeAllBasicInformation(object):

    DF_TEST = pd.DataFrame({'name': ['Earth', 'Moon', 'Mars', 'Mars', 'Moon'],
                            'mass_to_earth': [1, 0.606, 0.107, 0.107, 0.606]})
    COLUMNS = ['name', 'mass_to_earth']

    # Checking the type of objects returned by a function
    def test_df_basic_info(self, df=DF_TEST):

        # Arrange
        df_test_for_return_type = pd.DataFrame([0, 5])
        expected = type(None), type(df_test_for_return_type), type((0, 1))
        # Act
        info, describe, shape = dataframe_info.dataframe_all_basic_information(df)
        actual = type(info), type(describe), type(shape)
        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message

    # Checking the correctness of the returned value of the number of unique elements in the columns of the dataframe
    def test_number_unique_labels_in_dataframe_columns(self, df=DF_TEST, columns=COLUMNS):

        # Arrange
        expected = [3, 3]

        # Act
        actual_as_array = dataframe_info.number_unique_labels_in_dataframe_columns(df, columns).values
        actual_as_list = [actual_as_array[0], actual_as_array[1]]
        actual = actual_as_list

        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message
