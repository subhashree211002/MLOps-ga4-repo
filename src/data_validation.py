import pandas as pd

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates the input DataFrame for schema and missing values.
    """
    required_columns = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]

    # Check for missing columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for missing values
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    # Check for valid data types
    for col in required_columns[:-1]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric.")

    return True

