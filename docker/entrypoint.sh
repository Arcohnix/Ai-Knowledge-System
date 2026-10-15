#!/usr/bin/env bash
set -e

echo "[entrypoint] Initializing database and verifying connections..."
python -m scripts.init_db

echo "[entrypoint] Database ready. Starting server: $@"
exec "$@"
