import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from joblib import dump
from src.data_loader import load_data
from src.data_validation import validate_data

def train_model():
    """
    Train a simple Logistic Regression model on the Iris dataset.
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Validate
    df = pd.concat([X_train, y_train], axis=1)
    validate_data(df)

    #Train classifier model
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train,y_train)

    #Log training accuracy
    prediction_tr=mod_dt.predict(X_train)
    print('The training accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction_tr,y_train)))
    
    # Save model
    dump(mod_dt, "models/model.pkl")
    print("✅ Model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()

