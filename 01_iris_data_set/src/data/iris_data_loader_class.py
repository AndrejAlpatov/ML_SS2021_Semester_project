from sklearn import datasets
import pandas as pd


class IrisDataLoader:
    def load_iris_data_set(self):
        return self.X, self.y

    def __init__(self):
        self.iris = datasets.load_iris()
        self.X = pd.DataFrame(self.iris.data)
        self.y = pd.DataFrame(self.iris.target)
