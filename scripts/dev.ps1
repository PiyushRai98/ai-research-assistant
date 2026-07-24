# Launch the backend API and the Streamlit frontend for local development.
# Usage:  ./scripts/dev.ps1
# Requires:  pip install ".[ml,ui,dev]"
$ErrorActionPreference = "Stop"

Write-Host "Starting backend API on http://localhost:8000 ..." -ForegroundColor Green
Start-Process -NoNewWindow powershell -ArgumentList `
    "uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 3

Write-Host "Starting Streamlit frontend on http://localhost:8501 ..." -ForegroundColor Green
$env:API_BASE_URL = "http://localhost:8000"
streamlit run app/frontend/app.py --server.port 8501
