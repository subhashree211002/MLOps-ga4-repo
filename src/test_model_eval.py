import pytest
from src.evaluate import evaluate_model
from joblib import load

def test_model_evaluation(tmp_path):
    # Assuming model.pkl & test data are fetched via DVC
    model = load("models/model.pkl")
    accuracy = evaluate_model(model)
    assert 0 <= accuracy <= 1

