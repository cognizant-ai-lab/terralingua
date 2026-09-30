#!/usr/bin/env bash
# demo/local/run.sh — Build and launch the TerraLingua local demo container.
#
# Usage:
#   # Default: empty grid world demo
#   ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh
#
#   # Use the bundled Neuro-SAN HOCON network in the graph world
#   ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh --hocon
#
#   # Run in the social-graph world (agents own nodes, form/break connections
#   # at runtime). Combine with --hocon to seed the network from the bundled
#   # HOCON instead of the default topology.
#   ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh --dynamic_graph
#   ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh --hocon --dynamic_graph
#
#   # Use a custom HOCON file (path can be relative to repo root or absolute)
#   NEURO_SAN_HOCON=/path/to/network.hocon \
#       ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh
#
# Stop:
#   docker stop terralingua-demo
set -euo pipefail

IMAGE="terralingua-demo"
CONTAINER="terralingua-demo"
PORT="${API_PORT:-8765}"
DEFAULT_HOCON="demo/local/neuro_san_network.hocon"

USE_HOCON=0
USE_DYNAMIC_GRAPH=0
for arg in "$@"; do
    case "$arg" in
        --hocon) USE_HOCON=1 ;;
        --dynamic_graph) USE_DYNAMIC_GRAPH=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "[demo] ERROR: unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[demo] ERROR: ANTHROPIC_API_KEY is not set." >&2
    echo "[demo] Usage: ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER_ARGS="${TERRALINGUA_RUNNER_ARGS:-}"
DOCKER_EXTRA_ARGS=()

# Persist supervisord-managed service logs on the host so they survive
# container removal.
HOST_LOGS_DIR="$PROJECT_ROOT/logs_demo/local"
mkdir -p "$HOST_LOGS_DIR"
DOCKER_EXTRA_ARGS+=("-v" "$HOST_LOGS_DIR:/app/logs")

# HOCON mode is enabled if either --hocon was passed or NEURO_SAN_HOCON is set.
# When both are present, NEURO_SAN_HOCON wins (explicit path overrides bundled).
HOCON_SOURCE=""
if [[ -n "${NEURO_SAN_HOCON:-}" ]]; then
    HOCON_SOURCE="$NEURO_SAN_HOCON"
elif [[ "$USE_HOCON" -eq 1 ]]; then
    HOCON_SOURCE="$DEFAULT_HOCON"
fi

if [[ -n "$HOCON_SOURCE" ]]; then
    if [[ "$HOCON_SOURCE" = /* ]]; then
        HOCON_PATH="$HOCON_SOURCE"
    else
        HOCON_PATH="$PROJECT_ROOT/$HOCON_SOURCE"
    fi
    if [[ ! -f "$HOCON_PATH" ]]; then
        echo "[demo] ERROR: HOCON file not found: $HOCON_PATH" >&2
        exit 1
    fi
    HOCON_BASENAME="$(basename "$HOCON_PATH")"
    HOCON_MOUNT="/app/demo/local/$HOCON_BASENAME"
    DOCKER_EXTRA_ARGS+=("--mount" "type=bind,src=${HOCON_PATH},dst=${HOCON_MOUNT},readonly")
    RUNNER_ARGS="${RUNNER_ARGS} --graph.agent_network_hocon_path ${HOCON_MOUNT}"
    RUNNER_ARGS="${RUNNER_ARGS} --graph.agent_network_bidirectional_edges"
    RUNNER_ARGS="${RUNNER_ARGS} --genome sentence_directed --no-food_mechanism"
    RUNNER_ARGS="${RUNNER_ARGS} --init_food 0"
    RUNNER_ARGS="${RUNNER_ARGS} --max_message_length 400"
    echo "[demo] Neuro-SAN HOCON import enabled: $HOCON_PATH"
    echo "[demo] Container HOCON path: $HOCON_MOUNT"
else
    echo "[demo] Running default grid demo (pass --hocon to use the bundled agent network)."
fi

if [[ "$USE_DYNAMIC_GRAPH" -eq 1 ]]; then
    RUNNER_ARGS="${RUNNER_ARGS} --world_type social_graph"
    echo "[demo] Social-graph world enabled."
fi

if [[ -n "$RUNNER_ARGS" ]]; then
    echo "[demo] Runner args:$RUNNER_ARGS"
fi

# Another scenario folder and preset can be baked in: SCENARIO_DIR=scenarios/x PRESET=x ./demo/local/run.sh
echo "[demo] Building image (scenario ${SCENARIO_DIR:-scenarios/demo}, preset ${PRESET:-demo})…"
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" \
    --build-arg SCENARIO_DIR="${SCENARIO_DIR:-scenarios/demo}" \
    --build-arg PRESET="${PRESET:-demo}" \
    --build-arg EXP_NAME="${EXP_NAME:-demo}" \
    "$PROJECT_ROOT"
echo "[demo] Image built."

# Remove any previous container
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "[demo] Starting container…"
echo "[demo]   host logs dir: $HOST_LOGS_DIR → /app/logs"
DOCKER_RUN_CMD=(
    docker run -d
    --name "$CONTAINER"
    -p "${PORT}:${PORT}"
    -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    -e "TERRALINGUA_RUNNER_ARGS=$RUNNER_ARGS"
)
if [[ ${#DOCKER_EXTRA_ARGS[@]} -gt 0 ]]; then
    DOCKER_RUN_CMD+=("${DOCKER_EXTRA_ARGS[@]}")
fi
DOCKER_RUN_CMD+=(
    "$IMAGE"
)
"${DOCKER_RUN_CMD[@]}"

# Wait for the API server to be ready (up to 90s)
echo "[demo] Waiting for demo to be ready…"
for i in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/api/stats" &>/dev/null; then
        echo "[demo] Ready."
        break
    fi
    if [[ $i -eq 90 ]]; then
        echo "[demo] ERROR: Demo did not become ready in time." >&2
        echo "[demo] Check logs: docker logs $CONTAINER" >&2
        exit 1
    fi
    sleep 1
done

URL="http://localhost:${PORT}/auto-login"
echo ""
echo "[demo] ✓ Demo ready — open this URL in your browser:"
echo "[demo]   $URL"
echo ""
# Try to open a browser automatically (works on desktop environments)
xdg-open "$URL" &>/dev/null || open "$URL" &>/dev/null || true

echo ""
echo "[demo] Container running. Logs: docker logs -f $CONTAINER"
echo "[demo] Stop with:          docker stop $CONTAINER"
