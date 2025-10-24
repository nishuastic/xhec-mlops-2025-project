#!/bin/bash

# TODO: Use this file in your Dockerfile to run the services

uv run prefect server start --host 0.0.0.0 --port 4201 &
uv run uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
