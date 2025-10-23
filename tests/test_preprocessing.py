import pandas as pd

from src.modelling.config import CATEGORICAL_COLS, NUMERIC_COLS
from src.modelling.preprocessing import add_target, filter_outliers, prepare_features


def test_add_target_and_prepare_features():
    """Ensure preprocessing pipeline functions correctly on dummy data."""
    df = pd.DataFrame(
        {
            "Rings": [5, 10, 3],
            "Sex": ["M", "F", "I"],
            "Length": [0.5, 0.6, 0.4],
            "Diameter": [0.4, 0.45, 0.35],
            "Height": [0.1, 0.15, 0.08],
            "Whole weight": [0.2, 0.25, 0.15],
            "Shucked weight": [0.1, 0.12, 0.09],
            "Viscera weight": [0.05, 0.06, 0.04],
            "Shell weight": [0.07, 0.08, 0.05],
        }
    )

    df = add_target(df)
    assert "Age" in df.columns
    assert "Rings" not in df.columns

    df_filtered = filter_outliers(df)
    assert not df_filtered.empty

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df_filtered)
    assert not X_train.empty
    assert preprocessor is not None
    assert all(col in df_filtered.columns for col in NUMERIC_COLS + CATEGORICAL_COLS)
