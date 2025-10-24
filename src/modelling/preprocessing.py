from typing import Tuple

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.modelling.config import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Age = Rings + 1.5 and drop Rings"""
    df[TARGET] = df["Rings"] + 1.5
    df = df.drop(columns="Rings")
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply IQR filtering on numerical columns"""
    df_filtered = df.copy()
    for col in NUMERIC_COLS:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_filtered = df_filtered[
            (df_filtered[col].isnull())
            | ((df_filtered[col] >= lower) & (df_filtered[col] <= upper))
        ]
    logger.info(f"Filtered outliers: {df.shape} -> {df_filtered.shape}")
    return df_filtered


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Split data and create ColumnTransformer"""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )

    return X_train, X_test, y_train, y_test, preprocessor
