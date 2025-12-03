import numpy as np
import pytest

from ml.model import train_model
from schema import PredictPayload


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


@pytest.fixture()
def lower_payload():
    return PredictPayload(
        age=30,
        workclass='Self-emp-not-inc',
        fnlgt=200000,
        education='11th',
        education_num=7,
        marital_status='Never-married',
        occupation='Sales',
        relationship='Not-in-family',
        race='Black',
        sex='Women',
        capital_gain=0,
        capital_loss=0,
        hours_per_week=30,
        native_country='United-States',
    )


@pytest.fixture()
def higher_payload():
    return PredictPayload(
        age=35,
        workclass='Self-emp-not-inc',
        fnlgt=300000,
        education='Doctorate',
        education_num=16,
        marital_status='Married-civ-spouse',
        occupation='Prof-specialty',
        relationship='Husband',
        race='White',
        sex='Men',
        capital_gain=0,
        capital_loss=0,
        hours_per_week=60,
        native_country='United-States',
    )
