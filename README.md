<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## Running the App
All the following commands should be run from the project root.


1. Clone the repo
2. Ensure you have python 3.11 (use pyenv if not)
3. pip install uv
4. uv sync
5. uv run uvicorn src.web_service.main:app

Navigate to the given local host url /docs

Using Docker:

1. Clone the repo
2. docker build -f Dockerfile.app -t abaloneage:latest .
3. docker run -p 0.0.0.0:8001:8001 -p 0.0.0.0:4201:4201 abaloneage:latest

Navigate to the given localhost url /docs for uvicorn (8001)

To activate Prefect locally:
1. Open the Prefect UI to verify the deployment: uv run prefect server start
2. Open a new terminal, then run: uv run prefect work-pool create "abalone-training-pool" --type process
3. uv run prefect worker start --pool "abalone-training-pool"
4. Open a new terminal, then run: uv run prefect deploy --name abalone-retrain-weekly

To activate Prefect in docker:

This is done automatically, simply navigate to the Prefect URL displayed when using docker run (4201).

## Contributors

Martyna Harasym

João Silva

Maxim Ochterbeck

Nischay Parekh

Samyukt Sriram
