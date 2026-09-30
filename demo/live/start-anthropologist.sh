#!/usr/bin/env bash
# Wrapper that builds the terralingua-anthropologist command line from env vars.
# Run from supervisord; not meant to be invoked directly.
set -euo pipefail

set -a
# shellcheck source=/dev/null
source /app/.env
set +a

NO_AUDIT_FLAG=""
if [[ "${ANTHRO_NO_AUDIT,,}" == "true" ]]; then
    NO_AUDIT_FLAG="--no-audit"
fi

exec terralingua-anthropologist \
    --exp_name "${EXP_NAME}" \
    --model "${ANTHRO_MODEL}" \
    --window "${ANTHRO_WINDOW}" \
    --detect-interval "${ANTHRO_DETECT_INTERVAL}" \
    --dispatch-severity "${ANTHRO_DISPATCH_SEVERITY}" \
    ${NO_AUDIT_FLAG:+$NO_AUDIT_FLAG}
