from typing import Dict
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

from modelling.utils import save_pickle
from modelling.config import PIPELINE_PATH, RANDOM_STATE


def train_model(X_train, y_train, preprocessor, model_type: str = "GradientBoosting") -> Pipeline:
    """
    Train a regression model wrapped in a preprocessing pipeline.

    The function combines a preprocessing step (e.g., one-hot encoding)
    with a chosen regression estimator, fits the pipeline on the provided
    training data, and returns the trained pipeline.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
        preprocessor (ColumnTransformer): Transformer for preprocessing.
        model_type (str): Model type ("LinearRegression", "Ridge",
            "RandomForest", "GradientBoosting").

    Returns:
        Pipeline: Fitted sklearn pipeline containing both preprocessing
        and model steps.
    """
    logger.info(f"Training model: {model_type}")

    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    logger.success(f"Model '{model_type}' trained successfully.")
    return pipe


def evaluate_model(pipe: Pipeline, X_test, y_test) -> Dict[str, float]:
    """Evaluate model performance using MAE, RMSE, and R² metrics."""
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_artifacts(pipe: Pipeline) -> None:
    """Save the trained pipeline (preprocessor + model) as a pickle file."""
    save_pickle(pipe, PIPELINE_PATH)
    logger.info(f"Saved full pipeline to {PIPELINE_PATH}")
