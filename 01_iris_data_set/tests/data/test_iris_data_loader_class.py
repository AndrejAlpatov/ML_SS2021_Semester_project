import pytest
import numpy as np

from data.iris_data_loader_class import IrisDataLoader


@pytest.fixture
def iris_data_loader():
    # setup
    yield IrisDataLoader()
    # tear down


class TestLoadIrisDataSet(object):
    def test_load_iris_data_set(self, iris_data_loader):
        X, y = iris_data_loader.load_iris_data_set()
        assert 150 == len(X)
        assert 150 == len(y)

    def test_load_iris_data_set_2(self, iris_data_loader, mocker):
        def mock_load_iris_data_set():
            return np.random.normal(size=(140, 4)), np.random.normal(size=(140, 1))

        # replace method of an object:
        iris_data_loader.load_iris_data_set = mock_load_iris_data_set
        X, y = iris_data_loader.load_iris_data_set()
        assert 140 == len(X)
        assert 140 == len(y)
