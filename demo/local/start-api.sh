#!/usr/bin/env bash
# Wait for Postgres, seed the demo user (idempotent), then start the API.
# Run from supervisord.
set -euo pipefail

set -a
# shellcheck source=/dev/null
source /app/.env
set +a

for _ in $(seq 1 60); do
    if pg_isready -h "${POSTGRES_HOST:-127.0.0.1}" -p "${POSTGRES_PORT:-5432}" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

python demo/local/create_demo_db.py --api-key "${ANTHROPIC_API_KEY}"

exec terralingua-dashboard --port "${API_PORT:-8765}" --workers "${API_WORKERS:-4}" --dev
