import pytest
from src.evaluate import evaluate_model
from joblib import load

def test_model_evaluation(tmp_path):
    # Assuming model.pkl & test data are fetched via DVC
    model = load("models/model.pkl")

    df = pd.DataFrame({
        'sepal_length': [5.1, 4.9],
        'sepal_width': [3.5, 3.0],
        'petal_length': [1.4, 1.4],
        'petal_width': [0.2, 0.2],
        'species': ['setosa', 'setosa']
    })
    
    X = df.drop("species", axis=1)
    y = df["species"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    accuracy = evaluate_model(model)
    assert 0 <= accuracy <= 1

