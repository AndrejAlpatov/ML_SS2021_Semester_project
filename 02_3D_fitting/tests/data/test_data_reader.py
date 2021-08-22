import pytest

from data.data_reader import read
import data.data_reader


class TestRead(object):
    def test_read(self):
        df = read()
        actual = len(df)
        expected = data.data_reader.NROWS
        assert actual == expected

    def test_read_1(self):
        df = read()
        actual = df.columns.tolist()
        expected = data.data_reader.COLUMNS
        assert set(actual) == set(expected)
