import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from src.data_loader import load_data

def evaluate_model(model):
    """
    Evaluate a trained model on test data.
    """
    _, X_test, _, y_test = load_data()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

if __name__ == "__main__":
    model = load("models/model.pkl")
    acc = evaluate_model(model)
    print(f"🎯 Model Accuracy: {acc:.4f}")

