from sklearn import datasets
import pandas as pd


def load_iris_data_set():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)
    return X, y
