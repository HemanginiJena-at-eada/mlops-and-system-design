from src.transform import Transformer, balance_dataset
import pandas as pd

import pytest


@pytest.fixture
def sample_churn_data():
    """Creates a sample dataframe based on Churn_Modelling_train_test.csv schema."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 3, 4, 5],
            "CustomerId": [15634602, 15647311, 15619304, 15701354, 15737888],
            "Surname": ["Hargrave", "Hill", "Onio", "Boni", "Mitchell"],
            "CreditScore": [619, 608, 502, 699, 850],
            "Geography": ["France", "Spain", "France", "France", "Spain"],
            "Gender": [
                "Female",
                "Female",
                "Female",
                "Female",
                "Female",
            ],  # Keeping same to test OneHotEncoder drop='first' behavior cleanly
            "Age": [42.0, 41.0, 42.0, 39.0, 43.0],
            "Tenure": [2, 1, 8, 1, 2],
            "Balance": [0.00, 83807.86, 159660.80, 0.00, 125510.82],
            "NumOfProducts": [1, 1, 3, 2, 1],
            "HasCrCard": [1.0, 0.0, 1.0, 0.0, 1.0],
            "IsActiveMember": [1.0, 1.0, 0.0, 0.0, 1.0],
            "EstimatedSalary": [101348.88, 112542.58, 113931.57, 93826.63, 79084.10],
            "Exited": [1, 0, 1, 0, 0],  # 2 Exited, 3 Retained
        }
    )


def test_transformer(sample_churn_data):
    """Tests the transformer's column dropping and one-hot encoding on Churn data."""
    transformer = Transformer()

    # Mocking the constants imported from metadata.py for predictable testing
    transformer.drop_columns = ["RowNumber", "CustomerId", "Surname"]
    transformer.one_hot_encoding_columns = ["Geography", "Gender"]
    transformer.binary_variable_columns = []

    # We must patch the global ONE_HOT_ENCODE_COLUMNS used explicitly in transform.py's _one_hot_encoding
    import src.transform as transform

    transform.ONE_HOT_ENCODE_COLUMNS = ["Geography", "Gender"]
    transform.COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

    # Run the transformation
    transformed_df = transformer.transform(sample_churn_data)

    # Assertions for Dropped Columns
    for col in transformer.drop_columns:
        assert col not in transformed_df.columns, f"Failed to drop column: {col}"

    # Assertions for One-Hot Encoding
    assert (
        "Geography" not in transformed_df.columns
    ), "Original Geography column not dropped"
    assert "Gender" not in transformed_df.columns, "Original Gender column not dropped"

    # Check if the Geography_Spain column was created (France is dropped as first)
    assert (
        "Geography_Spain" in transformed_df.columns
    ), "Missing encoded column: Geography_Spain"


def test_balance_dataset(sample_churn_data):
    """Tests the dataset balancing logic using the 'Exited' target variable."""
    # In the sample fixture:
    # Class 1 (Exited=1) count = 2
    # Class 0 (Exited=0) count = 3
    # The balance_dataset function should downsample Class 0 to match Class 1.

    balanced_df = balance_dataset(sample_churn_data, target="Exited")

    # The minority class has 2 samples. Balanced size should be 2 + 2 = 4.
    assert len(balanced_df) == 4, f"Expected length 4, but got {len(balanced_df)}"

    # Verify exact distribution is 50/50
    counts = balanced_df["Exited"].value_counts()
    assert counts[0] == 2, "Class 0 was not properly balanced"
    assert counts[1] == 2, "Class 1 was altered unexpectedly"
