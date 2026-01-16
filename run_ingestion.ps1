# Run Data Ingestion Pipeline
# This script uses the venv Python directly to avoid conda conflicts

$env:PYTHONPATH = "d:\Projects\churn_predictor"
.\venv\Scripts\python.exe src/data/ingestion.py
