import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path: str = f"/home/subhashreemanogaran/MLOps-ga4-repo/data/data.csv"):
    """
    Load the Iris dataset from CSV and split into train/test.
    """
    df = pd.read_csv(data_path)

    if "species" not in df.columns:
        raise ValueError("Target column 'species' not found in dataset.")

    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
