#!/usr/bin/env bash
# Wait for Postgres, then start the API server. Run from supervisord.
set -euo pipefail

set -a
# shellcheck source=/dev/null
source /app/.env
set +a

# Wait up to ~30s for Postgres to accept connections. supervisord starts both
# postgres and api in parallel (priorities 5 and 20), so the api may race ahead.
for _ in $(seq 1 60); do
    if pg_isready -h "${POSTGRES_HOST:-127.0.0.1}" -p "${POSTGRES_PORT:-5432}" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

exec terralingua-dashboard --port "${API_PORT:-8765}" --workers "${API_WORKERS:-4}"
