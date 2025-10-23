import pandas as pd
from modelling.utils import load_pickle
from modelling.config import PIPELINE_PATH
from loguru import logger


def predict(new_data: pd.DataFrame) -> pd.Series:
    """Predict Abalone age given input features"""
    pipe = load_pickle(PIPELINE_PATH)
    preds = pipe.predict(new_data)
    logger.success("Prediction complete.")
    return preds