import pytest

from data.iris_util import load_iris_data_set

import numpy as np
from sklearn.utils import Bunch


class TestLoadIrisDataSet(object):
    def test_load_iris_data_set(self):
        X, y = load_iris_data_set()
        expected = 150
        actual = X.shape[0]
        message = "The size of the loaded Iris data set is expected to be {0} " \
                  "but was actual {1}".format(expected, actual)
        assert expected == actual, message

    def test_load_iris_data_set_2(self, mocker):
        """
        See also http://www.voidspace.org.uk/python/mock/patch.html#where-to-patch

        Mock where the object is imported into not where the object is imported from.

        :param mocker: the mocking environment
        :return: void
        """
        mocker.patch("data.iris_util.datasets.load_iris",
                     return_value=Bunch(data=np.random.normal(size=(140, 4)),
                                        target=np.random.normal(size=(140, 1))))
        X, y = load_iris_data_set()
        assert 140 == len(X)
        assert 140 == len(y)

    def test_load_iris_data_set_3(self, mocker):
        """
        See also https://changhsinlee.com/pytest-mock/

        :param mocker: the mocking environment
        :return: void
        """

        def mock_load_iris_data_set():
            return Bunch(data=np.random.normal(size=(130, 4)),
                         target=np.random.normal(size=(130, 1)))

        mocker.patch("data.iris_util.datasets.load_iris", mock_load_iris_data_set)
        X, y = load_iris_data_set()
        assert 130 == len(X)
        assert 130 == len(y)
