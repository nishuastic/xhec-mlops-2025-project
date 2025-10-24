from prefect import serve

from src.modelling.prefect_flows import train_flow

train_weekly = train_flow.to_deployment(
    name="abalone-retrain-weekly",
    version="0.1.0",
    tags=["training", "weekly", "retrain"],
    interval=7 * 24 * 60 * 60,
    parameters={
        "input_filepath": "data/abalone.csv",
        "artifacts_dirpath": "models",
        "model_type": "xgboost",
    },
)

if __name__ == "__main__":
    serve(train_weekly)
