#!/usr/bin/env bash
# Wrapper that builds the terralingua command line from env vars set in /app/.env.
# Run from supervisord; not meant to be invoked directly.
set -euo pipefail

set -a
# shellcheck source=/dev/null
source /app/.env
set +a

# Boolean flags: include when the env var is true, omit otherwise.
flag() {
    local val="${1:-}"
    local on="$2"
    local off="${3:-}"
    if [[ "${val,,}" == "true" ]]; then
        echo "$on"
    elif [[ -n "$off" ]]; then
        echo "$off"
    fi
}

# BooleanOptionalAction flags emit --foo or --no-foo.
REMOTE_API_FLAG=$(flag "${REMOTE_API_ENABLED:-true}" "--remote_api_enabled" "--no-remote_api_enabled")
SAVE_VIDEO_FLAG=$(flag "${SAVE_VIDEO:-false}" "--save_video" "--no-save_video")
RESUME_FLAG=$(flag "${RESUME:-false}" "--resume")

exec terralingua ${TERRALINGUA_PRESET:+"$TERRALINGUA_PRESET"} \
    "$REMOTE_API_FLAG" \
    "$SAVE_VIDEO_FLAG" \
    --exp_name "${EXP_NAME}" \
    --init_agents "${INIT_AGENTS}" \
    --min_agents "${MIN_AGENTS}" \
    --init_food "${INIT_FOOD}" \
    --grid_size "${GRID_SIZE}" \
    --food_zones "${FOOD_ZONES}" \
    --env_heartbeat "${ENV_HEARTBEAT}" \
    --max_ts "${MAX_TS}" \
    --empty_countdown "${EMPTY_COUNTDOWN}" \
    --max_parallel_workers "${MAX_PARALLEL_WORKERS}" \
    ${RESUME_FLAG:+$RESUME_FLAG}
