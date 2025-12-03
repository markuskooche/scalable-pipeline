import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics, inference


def test_train_model(x_train, y_train):
    model = train_model(x_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(y_train):
    precision, recall, f_beta = compute_model_metrics(y_train, y_train)
    assert np.round(precision, 2) == 1
    assert np.round(recall, 2) == 1
    assert np.round(f_beta, 2) == 1


def test_inference(model, x_test):
    preds = inference(model, x_test)
    assert isinstance(preds, np.ndarray)
