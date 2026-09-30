#!/usr/bin/env bash
# demo/live/docker-entrypoint.sh — live container entrypoint.
#
# Differences from the local demo:
#   - No DEMO_MODE, no /auto-login.
#   - No pre-seeded demo user; users sign up via /auth/register.
#   - /app/data is bind-mounted from the host; it holds the Postgres cluster
#     (pgdata/), simulation outputs (<EXP_NAME>/), and per-service logs
#     (run_logs/). The user database and simulation progress survive container
#     restarts.
set -euo pipefail

cd /app

# ── Require pinned secrets ──────────────────────────────────────────────────
# Auto-generating these would silently invalidate everyone's encrypted API keys
# and sessions on the next restart. For a persistent multi-user deployment we
# fail fast instead and force the operator to pin them in live.env (or an
# upstream secret manager).
MISSING_SECRETS=()
[[ -z "${FERNET_SECRET_KEY:-}" ]] && MISSING_SECRETS+=("FERNET_SECRET_KEY")
[[ -z "${JWT_SECRET:-}" ]]        && MISSING_SECRETS+=("JWT_SECRET")
[[ -z "${SESSION_SECRET:-}" ]]    && MISSING_SECRETS+=("SESSION_SECRET")
[[ -z "${POSTGRES_PASSWORD:-}" ]] && MISSING_SECRETS+=("POSTGRES_PASSWORD")
if (( ${#MISSING_SECRETS[@]} > 0 )); then
    echo "[entrypoint] ERROR: the following secret(s) are not set: ${MISSING_SECRETS[*]}" >&2
    echo "[entrypoint] Generate and pin them in demo/live/live.env:" >&2
    echo "[entrypoint]   FERNET_SECRET_KEY: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'" >&2
    echo "[entrypoint]   JWT_SECRET / SESSION_SECRET: python -c 'import secrets; print(secrets.token_hex(32))'" >&2
    echo "[entrypoint]   POSTGRES_PASSWORD: python -c 'import secrets; print(secrets.token_urlsafe(32))'" >&2
    echo "[entrypoint] Auto-generation would silently invalidate stored API keys and sessions on the next restart." >&2
    exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[entrypoint] Note: ANTHROPIC_API_KEY is not set. Each registered user supplies their own key after signing in; this is fine for the live demo." >&2
fi

# ── Postgres bootstrap ────────────────────────────────────────────────────
# Defaults baked into the image, overridable via env. POSTGRES_PASSWORD comes
# from the environment (live.env / secret manager); the role is created with it
# on first init.
POSTGRES_DB="${POSTGRES_DB:-ogw_users}"
POSTGRES_USER="${POSTGRES_USER:-ogw}"
POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
PG_BIN=$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)
PGDATA="${PGDATA:-/app/data/pgdata}"
RUN_LOGS_DIR="${TL_LOGS_DIR:-/app/data}/run_logs"

# Ensure run_logs/ exists in the bind-mounted dir before supervisord tries to
# write its program logs there.
mkdir -p "$RUN_LOGS_DIR"

# Bind-mounted host dirs come in owned by root (or the host user) and the
# Dockerfile-time mkdir under /app/data is hidden by the bind mount. Create
# the cluster dir if needed, then chown — Postgres refuses to run on a
# non-postgres-owned data dir.
mkdir -p "$PGDATA"
chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

FRESH_CLUSTER=0
if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    FRESH_CLUSTER=1
    echo "[entrypoint] Initializing Postgres cluster at $PGDATA…"
    # Local (Unix socket) → trust: lets the entrypoint create the role + DB
    # without juggling a superuser password. Host (TCP) → scram-sha-256: the
    # app connects over 127.0.0.1 with the role password.
    su postgres -c "$PG_BIN/initdb --auth-local=trust --auth-host=scram-sha-256 --username=postgres --no-locale --encoding=UTF8 -D $PGDATA" > /dev/null
    # Listen only on loopback; the cluster is private to the container.
    echo "listen_addresses = '127.0.0.1'" >> "$PGDATA/postgresql.conf"
    echo "unix_socket_directories = '/tmp'" >> "$PGDATA/postgresql.conf"
fi

if [[ "$FRESH_CLUSTER" == "1" ]]; then
    echo "[entrypoint] Creating role '$POSTGRES_USER' and database '$POSTGRES_DB'…"
    su postgres -c "$PG_BIN/pg_ctl -D $PGDATA -l /tmp/pg_init.log -o '-h 127.0.0.1 -p $POSTGRES_PORT' start" > /dev/null
    until PGPASSWORD=ignored su postgres -c "$PG_BIN/psql -h /tmp -p $POSTGRES_PORT -U postgres -d postgres -c 'SELECT 1' >/dev/null 2>&1"; do
        sleep 0.5
    done
    su postgres -c "$PG_BIN/psql -h /tmp -p $POSTGRES_PORT -U postgres -d postgres" <<SQL
CREATE ROLE $POSTGRES_USER WITH LOGIN PASSWORD '$POSTGRES_PASSWORD';
CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;
SQL
    su postgres -c "$PG_BIN/pg_ctl -D $PGDATA stop -m fast" > /dev/null
fi

DATABASE_URL="postgresql+asyncpg://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"

# Write env to /app/.env so supervisord-spawned child shells inherit it via
# `source /app/.env`. Includes the runner / anthropologist / API knobs so the
# start-*.sh wrappers can read them.
cat > /app/.env <<EOF
FERNET_SECRET_KEY=${FERNET_SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
SESSION_SECRET=${SESSION_SECRET}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
DATABASE_URL=${DATABASE_URL}

# API
API_PORT=${API_PORT:-8765}
API_HOST=${API_HOST:-0.0.0.0}
API_WORKERS=${API_WORKERS:-4}
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-*}

# Persistence
TL_LOGS_DIR=${TL_LOGS_DIR:-/app/data}
RUN_LOGS_DIR=${RUN_LOGS_DIR}
PGDATA=${PGDATA}
PG_BIN=${PG_BIN}
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}

# Experiment (consumed by start-runner.sh and start-anthropologist.sh)
EXP_NAME=${EXP_NAME:-live}
INIT_AGENTS=${INIT_AGENTS:-5}
MIN_AGENTS=${MIN_AGENTS:-5}
INIT_FOOD=${INIT_FOOD:-500}
GRID_SIZE=${GRID_SIZE:-50}
FOOD_ZONES=${FOOD_ZONES:-3}
ENV_HEARTBEAT=${ENV_HEARTBEAT:-20}
MAX_TS=${MAX_TS:--1}
EMPTY_COUNTDOWN=${EMPTY_COUNTDOWN:--1}
MAX_PARALLEL_WORKERS=${MAX_PARALLEL_WORKERS:-64}
RESUME=${RESUME:-true}
REMOTE_API_ENABLED=${REMOTE_API_ENABLED:-true}
SAVE_VIDEO=${SAVE_VIDEO:-false}

# Anthropologist
ANTHRO_WINDOW=${ANTHRO_WINDOW:-10}
ANTHRO_DETECT_INTERVAL=${ANTHRO_DETECT_INTERVAL:-3}
ANTHRO_DISPATCH_SEVERITY=${ANTHRO_DISPATCH_SEVERITY:-1}
ANTHRO_NO_AUDIT=${ANTHRO_NO_AUDIT:-true}
ANTHRO_MODEL=${ANTHRO_MODEL:-claude-haiku-4-5}
EOF

mkdir -p "$TL_LOGS_DIR" "$RUN_LOGS_DIR"

echo "[entrypoint] Starting services via supervisord…"
exec supervisord -n -c /app/demo/live/supervisord.conf
