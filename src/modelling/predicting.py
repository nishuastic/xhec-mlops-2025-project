import pandas as pd
from loguru import logger

from src.modelling.config import PIPELINE_PATH
from src.modelling.utils import load_pickle


def predict(new_data: pd.DataFrame) -> pd.Series:
    """Predict Abalone age given input features"""
    pipe = load_pickle(PIPELINE_PATH)
    preds = pipe.predict(new_data)
    logger.success("Prediction complete.")
    return preds
