import pandas as pd
from src.modelling.training import train_model
from src.modelling.preprocessing import prepare_features
from src.modelling.utils import save_pickle
from src.modelling.predicting import predict
from src.modelling.config import PIPELINE_PATH


def test_predict_function(tmp_path):
    """Ensure predict() works correctly on trained pipeline."""
    df = pd.DataFrame({
        "Sex": ["M", "F", "I"],
        "Length": [0.5, 0.6, 0.4],
        "Diameter": [0.4, 0.45, 0.35],
        "Height": [0.1, 0.15, 0.08],
        "Whole weight": [0.2, 0.25, 0.15],
        "Shucked weight": [0.1, 0.12, 0.09],
        "Viscera weight": [0.05, 0.06, 0.04],
        "Shell weight": [0.07, 0.08, 0.05],
        "Age": [6.5, 11.5, 4.5],
    })

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)
    pipe = train_model(X_train, y_train, preprocessor)
    save_pickle(pipe, PIPELINE_PATH)

    preds = predict(X_test)
