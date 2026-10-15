#!/usr/bin/env bash
set -e

# Load environment variables if .env exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Starting AI Knowledge System development server on http://$HOST:$PORT..."
uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
