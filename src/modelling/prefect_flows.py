from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from prefect import flow, get_run_logger, task

from .config import DATA_DIR, MODEL_TYPE
from .preprocessing import add_target, filter_outliers, load_data, prepare_features
from .training import evaluate_model, save_artifacts, train_model


@task
def t_load_data(path: str) -> pd.DataFrame:
    """Load data from CSV."""
    return load_data(path)


@task
def t_add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variable."""
    return add_target(df)


@task
def t_filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers from data."""
    return filter_outliers(df)


@task
def t_prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Prepare features and split data."""
    return prepare_features(df)


@task
def t_train_model(X_train, y_train, preprocessor, model_type: str):
    """Train regression model."""
    return train_model(X_train, y_train, preprocessor, model_type=model_type)


@task
def t_evaluate_model(pipe, X_test, y_test) -> Dict[str, float]:
    """Evaluate model performance."""
    return evaluate_model(pipe, X_test, y_test)


@task
def t_save_artifacts(pipe, artifacts_dirpath: str) -> str:
    """Save model artifacts."""
    os.makedirs(artifacts_dirpath, exist_ok=True)
    save_artifacts(pipe)
    return str(Path(artifacts_dirpath).resolve())


@flow(name="abalone-train-flow")
def train_flow(
    input_filepath: Optional[str] = None,
    artifacts_dirpath: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Dict[str, object]:
    """Run the full training flow."""
    logger = get_run_logger()

    if input_filepath is None:
        input_filepath = str(DATA_DIR / "abalone.csv")
    if artifacts_dirpath is None:
        artifacts_dirpath = str(Path.cwd() / "models")
    if model_type is None:
        model_type = MODEL_TYPE

    logger.info(f"Start Prefect flow; model={model_type}")

    df = t_load_data.submit(input_filepath)
    df = t_add_target.submit(df)
    df = t_filter_outliers.submit(df)
    X_train, X_test, y_train, y_test, preprocessor = t_prepare_features.submit(
        df
    ).result()
    pipe = t_train_model.submit(X_train, y_train, preprocessor, model_type).result()
    metrics = t_evaluate_model.submit(pipe, X_test, y_test).result()
    artifacts_path = t_save_artifacts.submit(pipe, artifacts_dirpath).result()

    logger.info(
        f"Done. RMSE={metrics['RMSE']:.3f} R2={metrics['R2']:.3f}; artifacts: {artifacts_path}"
    )

    return {"metrics": metrics, "artifacts_dir": artifacts_path}


if __name__ == "__main__":
    train_flow()
