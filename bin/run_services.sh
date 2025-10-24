#!/bin/bash
echo "🚀 Starting Prefect server..."
uv run prefect server start --host 0.0.0.0 --port 4201 &

# Wait for the Prefect API to be available
echo "⏳ Waiting for Prefect API to become available..."
sleep 10


# Create or get the work pool
uv run prefect work-pool ls | grep "abalone-training-pool" || \
  uv run prefect work-pool create "abalone-training-pool" --type process

# Start worker (run in background)
echo "👷 Starting Prefect worker..."
uv run prefect worker start --pool "abalone-training-pool" &

# Deploy flow (assuming deployment spec exists in prefect.yaml)
echo "📦 Deploying abalone retrain flow..."
uv run prefect deploy --name abalone-retrain-weekly || echo "⚠️ Deployment failed (maybe already deployed)."

# Start FastAPI web service
echo "🌐 Starting web service..."
uv run uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
