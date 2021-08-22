import pytest
import numpy as np
import pandas as pd

from features.scaling_util import scale_feature
from data.iris_util import load_iris_data_set


@pytest.fixture
def data_set():
    # setup
    yield pd.DataFrame(np.random.normal(size=(150, 4)))
    # tear down


class TestScaleFeature(object):
    def test_scale_feature(self):
        expected_mean = 0.
        X, y = load_iris_data_set()
        X_scaled = scale_feature(X)
        actual_features = X_scaled.mean(axis=0)
        for i in range(4):
            message = 'Expected 0. but got a mean after standard scaling' \
                      ' of {0}'.format(actual_features[i])
            assert expected_mean == pytest.approx(actual_features[i]), message

    def test_scale_feature_with_random_normal_data(self, data_set):
        expected_mean = 0.
        X = data_set
        X_scaled = scale_feature(X)
        actual_features = X_scaled.mean(axis=0)
        for i in range(4):
            message = 'Expected 0. but got a mean after standard scaling' \
                      ' of {0}'.format(actual_features[i])
            assert expected_mean == pytest.approx(actual_features[i]), message
