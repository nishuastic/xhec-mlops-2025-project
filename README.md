# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)

## Quick Start

### Local Development

```bash
# Install dependencies
pip install uv
uv sync

# Run the web service
uv run uvicorn src.web_service.main:app
```

Navigate to `http://localhost:8000/docs` for the API documentation.

### Docker

```bash
# Build and run
docker build -f Dockerfile.app -t abaloneage:latest .
docker run -p 8001:8001 -p 4201:4201 abaloneage:latest
```

- API docs: `http://localhost:8001/docs`
- Prefect UI: `http://localhost:4201`

### Prefect Workflows

**Local:**
```bash
# Start Prefect server
uv run prefect server start

# Create work pool (new terminal)
uv run prefect work-pool create "abalone-training-pool" --type process

# Start worker (new terminal)
uv run prefect worker start --pool "abalone-training-pool"

# Deploy workflow (new terminal)
uv run prefect deploy --name abalone-retrain-weekly
```

**Docker:**
Prefect is automatically configured. Access the Prefect UI at the displayed URL (port 4201).

## Contributors

Martyna Harasym

João Silva

Maxim Ochterbeck

Nischay Parekh

Samyukt Sriram
