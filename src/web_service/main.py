# Code with FastAPI (app = FastAPI(...)


import pandas as pd
from fastapi import FastAPI

from src.modelling.config import PIPELINE_PATH

# Other imports
from src.web_service.lib.models import AbaloneInput, PredObj
from src.web_service.utils import load_pickle

app = FastAPI(
    title="Abalone-Size-Predictor", description="Predicts the size of Abalones"
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredObj, status_code=201)
def predict(payload: AbaloneInput) -> PredObj:
    # TODO: complete and replace the "InsertHereAPydanticClass" with the correct Pydantic classes defined in web_service/lib/models.py
    df = pd.DataFrame([payload.dict(by_alias=True)])
    pipe = load_pickle(PIPELINE_PATH)
    preds = pipe.predict(df)[0]
    return {"Age": preds}
