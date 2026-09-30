#!/usr/bin/env bash
# demo/live/run.sh — Build and launch the TerraLingua live demo container.
#
# Setup:
#   cp demo/live/live.env.example demo/live/live.env
#   # fill in the secrets in demo/live/live.env (FERNET/JWT/SESSION/POSTGRES_PASSWORD)
#
# Usage:
#   ./demo/live/run.sh                # use the EXP_NAME / RESUME baked into the image
#   ./demo/live/run.sh --resume       # resume from the latest checkpoint
#   ./demo/live/run.sh --no-resume    # force a fresh run
#   ./demo/live/run.sh --exp NAME     # run/resume experiment NAME
#
# EXP_NAME, RESUME and the rest of the non-secret config default to the
# Dockerfile's ENV; --exp / --resume pass `-e` overrides without rebuilding.
#
# Stop:
#   docker stop terralingua-live
set -euo pipefail

IMAGE="terralingua-live"
CONTAINER="terralingua-live"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/live.env"

# Optional CLI overrides, passed to the container via `-e` (which beats the
# baked-in ENV). Leave unset to use the Dockerfile defaults.
RESUME_OVERRIDE=""
EXP_NAME_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)    RESUME_OVERRIDE="true" ;;
        --no-resume) RESUME_OVERRIDE="false" ;;
        --exp)       EXP_NAME_OVERRIDE="${2:?--exp requires a name}"; shift ;;
        --exp=*)     EXP_NAME_OVERRIDE="${1#--exp=}" ;;
        *) echo "[live] ERROR: unknown argument: $1 (expected --resume/--no-resume/--exp NAME)" >&2; exit 1 ;;
    esac
    shift
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[live] ERROR: $ENV_FILE not found." >&2
    echo "[live] Copy $SCRIPT_DIR/live.env.example to live.env and edit it." >&2
    exit 1
fi

# Source the secrets file for the host-side knobs (HOST_DATA_PATH, optional
# ANTHROPIC_API_KEY). API_PORT defaults to the Dockerfile's 8765 unless you also
# set it here. The file is still passed to the container via --env-file below.
# shellcheck source=/dev/null
set -a; source "$ENV_FILE"; set +a

if [[ "${ANTHROPIC_API_KEY:-}" == "sk-ant-..." ]]; then
    # Treat the placeholder as unset.
    unset ANTHROPIC_API_KEY
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[live] Note: ANTHROPIC_API_KEY is not set in $ENV_FILE. Each registered user will supply their own key from the dashboard." >&2
fi

PORT="${API_PORT:-8765}"
HOST_DATA_PATH="${HOST_DATA_PATH:-$PROJECT_ROOT/logs_demo/live}"

# Resolve to absolute path (docker -v requires it).
mkdir -p "$HOST_DATA_PATH"
HOST_DATA_PATH="$(cd "$HOST_DATA_PATH" && pwd)"

echo "[live] Building image…"
docker build --target local -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
echo "[live] Image built."

# Remove any previous container.
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "[live] Starting container…"
echo "[live]   data dir: $HOST_DATA_PATH → /app/data"
echo "[live]     ├── <EXP_NAME>/   simulation checkpoints, agent logs, field notes"
echo "[live]     ├── run_logs/     supervisord-managed service logs"
echo "[live]     └── pgdata/       Postgres cluster"

docker run -d \
    --name "$CONTAINER" \
    -p "${PORT}:${PORT}" \
    --env-file "$ENV_FILE" \
    ${RESUME_OVERRIDE:+-e RESUME=$RESUME_OVERRIDE} \
    ${EXP_NAME_OVERRIDE:+-e EXP_NAME=$EXP_NAME_OVERRIDE} \
    -v "$HOST_DATA_PATH:/app/data" \
    --stop-timeout 90 \
    "$IMAGE"

echo "[live] Waiting for API to be ready…"
for i in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/api/stats" &>/dev/null; then
        echo "[live] Ready."
        break
    fi
    if [[ $i -eq 90 ]]; then
        echo "[live] ERROR: API did not become ready in time." >&2
        echo "[live] Check logs: docker logs $CONTAINER" >&2
        exit 1
    fi
    sleep 1
done

echo ""
echo "[live] ✓ Live demo running on http://localhost:${PORT}"
echo "[live]   Sign up:  http://localhost:${PORT}/register"
echo "[live]   Sign in:  http://localhost:${PORT}/login"
echo ""
echo "[live] Logs:  docker logs -f $CONTAINER"
echo "[live] Stop:  docker stop $CONTAINER"
