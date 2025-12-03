import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    x_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    f_beta : float
    """
    f_beta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, f_beta


def inference(model, x):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    x : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(x)

def compute_sliced_model_metrics(model, data, cat_features=[], label=None, encoder=None, lb=None):
    """
    Compute model metrics on slices of the data.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    data : np.ndarray
        Data used for testing.
    cat_features : list[str]
        List of categorical features.
    label : str
        Name of the label column.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    metrics : pd.DataFrame
        Model metrics for the slice of data.
    """
    metrics = []

    for feature in cat_features:
        for value in data[feature].unique():
            x_loc = data.loc[data[feature] == value]
            x, y, encoder, lb = process_data(x_loc, cat_features, label, False, encoder, lb)

            preds = inference(model, x)
            precision, recall, f_beta = compute_model_metrics(y, preds)
            metrics.append((feature, value, precision, recall, f_beta))

    return pd.DataFrame(metrics, columns=['feature', 'value', 'precision', 'recall', 'fbeta'])
