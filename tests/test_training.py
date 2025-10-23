import pandas as pd
from src.modelling.training import train_model, evaluate_model
from src.modelling.preprocessing import prepare_features
from src.modelling.config import MODEL_TYPE


def test_train_and_evaluate_model():
    """Train and evaluate model on dummy dataset."""
    df = pd.DataFrame({
        "Sex": ["M", "F", "I", "M"],
        "Length": [0.5, 0.6, 0.4, 0.55],
        "Diameter": [0.4, 0.45, 0.35, 0.42],
        "Height": [0.1, 0.15, 0.08, 0.12],
        "Whole weight": [0.2, 0.25, 0.15, 0.22],
        "Shucked weight": [0.1, 0.12, 0.09, 0.11],
        "Viscera weight": [0.05, 0.06, 0.04, 0.05],
        "Shell weight": [0.07, 0.08, 0.05, 0.07],
        "Age": [6.5, 11.5, 4.5, 7.5],
    })

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)
    pipe = train_model(X_train, y_train, preprocessor, model_type=MODEL_TYPE)
    metrics = evaluate_model(pipe, X_test, y_test)

    assert "RMSE" in metrics
    assert isinstance(metrics["R2"], float)
