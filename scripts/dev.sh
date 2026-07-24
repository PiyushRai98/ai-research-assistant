#!/usr/bin/env bash
# Launch the backend API and the Streamlit frontend for local development.
# Usage:  ./scripts/dev.sh
# Requires:  pip install ".[ml,ui,dev]"
set -euo pipefail

echo "Starting backend API on http://localhost:8000 ..."
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Stop the backend when this script exits.
trap 'kill "${BACKEND_PID}" 2>/dev/null || true' EXIT

sleep 3

echo "Starting Streamlit frontend on http://localhost:8501 ..."
API_BASE_URL="http://localhost:8000" \
    streamlit run app/frontend/app.py --server.port 8501
