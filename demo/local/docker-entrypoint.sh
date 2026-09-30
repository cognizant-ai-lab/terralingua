#!/usr/bin/env bash
set -euo pipefail

cd /app

# ── Generate secrets if not provided ─────────────────────────────────────────
if [[ -z "${FERNET_SECRET_KEY:-}" ]]; then
    FERNET_SECRET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    export FERNET_SECRET_KEY
fi

if [[ -z "${JWT_SECRET:-}" ]]; then
    JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
    export JWT_SECRET
fi

if [[ -z "${SESSION_SECRET:-}" ]]; then
    SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
    export SESSION_SECRET
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[entrypoint] ERROR: ANTHROPIC_API_KEY is required." >&2
    exit 1
fi

# ── Postgres bootstrap ───────────────────────────────────────────────────────
# Local demo: PGDATA lives inside the container (no bind mount) so it's
# ephemeral. We initdb on every fresh container launch.
POSTGRES_DB="${POSTGRES_DB:-ogw_users}"
POSTGRES_USER="${POSTGRES_USER:-ogw}"
POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
PG_BIN=$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)
PGDATA="${PGDATA:-/app/pgdata}"

chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    echo "[entrypoint] Initializing Postgres cluster at $PGDATA…"
    # Local (Unix socket) trust + host (TCP) scram-sha-256: avoids needing
    # a superuser password while still requiring auth over the loopback port.
    su postgres -c "$PG_BIN/initdb --auth-local=trust --auth-host=scram-sha-256 --username=postgres --no-locale --encoding=UTF8 -D $PGDATA" > /dev/null
    echo "listen_addresses = '127.0.0.1'" >> "$PGDATA/postgresql.conf"
    echo "unix_socket_directories = '/tmp'" >> "$PGDATA/postgresql.conf"
fi

echo "[entrypoint] Creating role '$POSTGRES_USER' and database '$POSTGRES_DB'…"
su postgres -c "$PG_BIN/pg_ctl -D $PGDATA -l /tmp/pg_init.log -o '-h 127.0.0.1 -p $POSTGRES_PORT' start" > /dev/null
until su postgres -c "$PG_BIN/psql -h /tmp -p $POSTGRES_PORT -U postgres -d postgres -c 'SELECT 1' >/dev/null 2>&1"; do
    sleep 0.5
done
su postgres -c "$PG_BIN/psql -h /tmp -p $POSTGRES_PORT -U postgres -d postgres" <<SQL
CREATE ROLE $POSTGRES_USER WITH LOGIN PASSWORD '$POSTGRES_PASSWORD';
CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;
SQL
su postgres -c "$PG_BIN/pg_ctl -D $PGDATA stop -m fast" > /dev/null

DATABASE_URL="postgresql+asyncpg://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"

cat > /app/.env <<EOF
FERNET_SECRET_KEY=${FERNET_SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
SESSION_SECRET=${SESSION_SECRET}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
DATABASE_URL=${DATABASE_URL}
DEMO_MODE=true
DEMO_USER_EMAIL=${DEMO_USER_EMAIL:-demo@demo.com}
DEMO_USER_NAME=${DEMO_USER_NAME:-demo}
DEMO_USER_PASS=${DEMO_USER_PASS:-demopass123}

# API
API_PORT=${API_PORT:-8765}
API_HOST=${API_HOST:-0.0.0.0}
API_WORKERS=${API_WORKERS:-4}
FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-127.0.0.1}

# Postgres
PGDATA=${PGDATA}
PG_BIN=${PG_BIN}
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
EOF

echo "[entrypoint] Starting services via supervisord…"
exec supervisord -n -c /app/demo/local/supervisord.conf
