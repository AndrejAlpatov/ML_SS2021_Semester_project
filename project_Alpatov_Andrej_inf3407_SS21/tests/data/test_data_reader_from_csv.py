import sys
sys.path.append('../../src/data/')
import data_reader_from_csv


class TestReadDataSet(object):

    # The check matches the number of lines read from the date set with the number in the created dataframe
    def test_read_data_set(self):

        # Arrange
        expected = data_reader_from_csv.NROWS

        # Act
        df = data_reader_from_csv.read_data_set()
        actual = len(df)
        error_message = "Expected: {0}, but Actual: {1}".format(expected, actual)

        # Assert
        assert actual == expected, error_message






