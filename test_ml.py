import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label
from ml.model import train_model, compute_model_metrics


def test_apply_labels():
    """
    Test that lavels are converted correctly
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"

def test_train_model():
    """
    Test that the model is trained correctly
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y)
    assert model is not None


def test_compute_model_metrics():
    """
    Test the model metrics
    """
    y = np.array([0, 0, 1, 1])
    preds = np.array([1, 0, 0, 0])
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert precision is not None
    assert recall is not None
    assert f1 is not None
