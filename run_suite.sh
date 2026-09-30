#!/usr/bin/env bash
# run_suite.sh — Launch the full TerraLingua suite locally (no Docker).
#
# Brings up: redis (if missing), terralingua-dashboard, terralingua, terralingua-anthropologist
# inside a tmux session, with each service's output tee'd to logs/<service>.log.
#
# Usage:
#   ./run_suite.sh            # bring everything up (default)
#   ./run_suite.sh attach     # attach to the running tmux session
#   ./run_suite.sh stop       # kill the tmux session (and redis, if we started it)
#   ./run_suite.sh status     # show window list + redis ping
#
# Common env overrides:
#   EXP_NAME=myrun            # shared exp name for runner + anthropologist  (default: dev)
#   API_PORT=8765             # API server port                              (default: 8765)
#   API_WORKERS=1             # uvicorn workers                              (default: 1)
#   REDIS_PORT=6379           # local redis port                             (default: 6379)
#   ANTHRO_MODEL=...          # anthropologist LLM model                     (default: claude-haiku-4-5)
#   ANTHRO_DETECT_INTERVAL=3  # detection cadence in steps                   (default: 3)
#   RUNNER_ARGS="..."         # extra args appended to terralingua
#   API_ARGS="..."            # extra args appended to terralingua-dashboard
#   ANTHRO_ARGS="..."         # extra args appended to terralingua-anthropologist
#   TMUX_SESSION=terralingua  # tmux session name                            (default: terralingua)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION="${TMUX_SESSION:-terralingua}"
LOG_DIR="$SCRIPT_DIR/logs"
EXP_NAME="${EXP_NAME:-dev}"
API_PORT="${API_PORT:-8765}"
API_WORKERS="${API_WORKERS:-1}"
REDIS_PORT="${REDIS_PORT:-6379}"
ANTHRO_MODEL="${ANTHRO_MODEL:-claude-haiku-4-5}"
ANTHRO_DETECT_INTERVAL="${ANTHRO_DETECT_INTERVAL:-3}"
RUNNER_ARGS="${RUNNER_ARGS:-}"
API_ARGS="${API_ARGS:-}"
ANTHRO_ARGS="${ANTHRO_ARGS:-}"

REDIS_PID_FILE="$LOG_DIR/redis.pid"
REDIS_OWNED_MARKER="$LOG_DIR/redis.owned"

# Each tmux pane runs this preamble so it picks up .env and our REDIS_URL.
PANE_PREAMBLE="cd '$SCRIPT_DIR'; [ -f .env ] && { set -a; . ./.env; set +a; }; export REDIS_URL='redis://localhost:$REDIS_PORT';"

_require() {
    command -v "$1" >/dev/null 2>&1 || { echo "[suite] ERROR: '$1' not found in PATH." >&2; exit 1; }
}

_start_redis_if_needed() {
    if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
        echo "[suite] redis already running on :$REDIS_PORT (reusing)"
        return
    fi
    echo "[suite] starting redis on :$REDIS_PORT"
    redis-server \
        --daemonize yes \
        --port "$REDIS_PORT" \
        --logfile "$LOG_DIR/redis.log" \
        --pidfile "$REDIS_PID_FILE" \
        --dir "$LOG_DIR"
    for _ in $(seq 1 20); do
        redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1 && break
        sleep 0.5
    done
    redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1 \
        || { echo "[suite] ERROR: redis failed to start; see $LOG_DIR/redis.log" >&2; exit 1; }
    touch "$REDIS_OWNED_MARKER"
}

_stop_redis_if_owned() {
    [[ -f "$REDIS_OWNED_MARKER" ]] || { echo "[suite] redis was already running before us — leaving it alone"; return; }
    if [[ -f "$REDIS_PID_FILE" ]]; then
        local pid; pid=$(cat "$REDIS_PID_FILE" 2>/dev/null || true)
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "[suite] redis stopped (pid $pid)"
        fi
        rm -f "$REDIS_PID_FILE"
    fi
    rm -f "$REDIS_OWNED_MARKER"
}

_pane_cmd() {
    # Wrap a command so the pane sources .env, tees to a log, and stays open on exit.
    local log="$1"; shift
    local cmd="$*"
    echo "$PANE_PREAMBLE { $cmd; } 2>&1 | tee -a '$log'; echo; echo '[$log] process exited — Ctrl-d to close pane.'; exec bash"
}

cmd="${1:-up}"
case "$cmd" in
    up|start)
        _require tmux
        _require redis-server
        _require redis-cli
        _require terralingua
        _require terralingua-dashboard
        _require terralingua-anthropologist

        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[suite] tmux session '$SESSION' already exists." >&2
            echo "[suite]   attach: ./run_suite.sh attach" >&2
            echo "[suite]   stop:   ./run_suite.sh stop" >&2
            exit 1
        fi

        mkdir -p "$LOG_DIR"
        _start_redis_if_needed

        API_CMD="terralingua-dashboard --port $API_PORT --workers $API_WORKERS $API_ARGS"
        RUNNER_CMD="terralingua --exp_name $EXP_NAME --remote_api_enabled $RUNNER_ARGS"
        ANTHRO_CMD="terralingua-anthropologist --exp_name $EXP_NAME --model $ANTHRO_MODEL --detect-interval $ANTHRO_DETECT_INTERVAL $ANTHRO_ARGS"

        tmux new-session -d -s "$SESSION" -n api            "$(_pane_cmd "$LOG_DIR/api.log"            "$API_CMD")"
        tmux new-window     -t "$SESSION" -n runner         "$(_pane_cmd "$LOG_DIR/runner.log"         "$RUNNER_CMD")"
        tmux new-window     -t "$SESSION" -n anthropologist "$(_pane_cmd "$LOG_DIR/anthropologist.log" "$ANTHRO_CMD")"
        tmux select-window  -t "$SESSION":api

        cat <<EOF

[suite] ✓ tmux session '$SESSION' started.
[suite]   attach:  tmux attach -t $SESSION   (or ./run_suite.sh attach)
[suite]   detach:  Ctrl-b d
[suite]   switch:  Ctrl-b 0/1/2  (api / runner / anthropologist)
[suite]   stop:    ./run_suite.sh stop

[suite] Logs:
[suite]   tail -f $LOG_DIR/api.log
[suite]   tail -f $LOG_DIR/runner.log
[suite]   tail -f $LOG_DIR/anthropologist.log

[suite] Dashboard: http://localhost:$API_PORT/dashboard
[suite] Experiment name (shared by runner + anthropologist): $EXP_NAME
EOF
        ;;

    attach)
        tmux attach -t "$SESSION"
        ;;

    stop|down)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            tmux kill-session -t "$SESSION"
            echo "[suite] tmux session '$SESSION' killed"
        else
            echo "[suite] no tmux session named '$SESSION'"
        fi
        _stop_redis_if_owned
        ;;

    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[suite] tmux windows:"
            tmux list-windows -t "$SESSION"
        else
            echo "[suite] no tmux session named '$SESSION'"
        fi
        if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            echo "[suite] redis :$REDIS_PORT — PONG"
        else
            echo "[suite] redis :$REDIS_PORT — not reachable"
        fi
        ;;

    *)
        echo "usage: $0 [up|attach|stop|status]" >&2
        exit 1
        ;;
esac
