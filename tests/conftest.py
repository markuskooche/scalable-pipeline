import numpy as np
import pytest

from ml.model import train_model


@pytest.fixture()
def x_train():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture()
def y_train():
    return np.array([1, 0, 1])


@pytest.fixture()
def model(x_train, y_train):
    return train_model(x_train, y_train)


@pytest.fixture()
def x_test():
    return np.array([[1, 2, 3]])
