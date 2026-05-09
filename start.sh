#!/bin/bash
# Entrypoint for both local Docker and Render deployments.
# Runs DB migrations, then starts the Celery worker in the background
# and Uvicorn in the foreground (single-container free-tier friendly).
set -e

echo "==> Running database migrations..."
python migrate.py

echo "==> Starting Celery worker (background)..."
celery -A app.celery_app worker --loglevel=info --concurrency=2 &

echo "==> Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
