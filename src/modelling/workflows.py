import os
from typing import Optional, Dict

from loguru import logger
from modelling.preprocessing import load_data, add_target, filter_outliers, prepare_features
from modelling.training import train_model, evaluate_model, save_artifacts
from modelling.config import DATA_DIR, MODEL_TYPE


def train_workflow(
    input_filepath: Optional[str] = None,
    artifacts_dirpath: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Dict[str, object]:
    """
    Complete training workflow:
    1. Load and preprocess data
    2. Train model
    3. Evaluate performance
    4. Save artifacts (model + preprocessor)

    Args:
        input_filepath (str): Path to input CSV dataset.
        artifacts_dirpath (str): Directory to save model artifacts.
        model_type (str): Model type to train ("Ridge", "RandomForest", ...)

    Returns:
        dict: metrics and trained model
    """
    if input_filepath is None:
        input_filepath = DATA_DIR / "abalone.csv"
    if artifacts_dirpath is None:
        artifacts_dirpath = os.path.join(os.getcwd(), "models")
    if model_type is None:
        model_type = MODEL_TYPE

    logger.info(f"Starting training workflow with model={model_type}")

    df = load_data(input_filepath)
    df = add_target(df)
    df = filter_outliers(df)
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)
    pipe = train_model(X_train, y_train, preprocessor, model_type=model_type)
    metrics = evaluate_model(pipe, X_test, y_test)
    save_artifacts(pipe)

    logger.success(
        f"Workflow complete! Model={model_type}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}"
    )

    return {"model": pipe, "metrics": metrics}
