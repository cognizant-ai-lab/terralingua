# syntax=docker/dockerfile:1.7
# demo/live/Dockerfile — TerraLingua live demo image (single, multi-stage).
#
# Two build targets share one `base` stage (system deps + all non-secret
# config as ENV). They differ only in how the source tree gets in:
#
#   • local      — COPYs the working tree. No git clone; for local testing.
#   • standalone — clones the public repository at build time.
# EXP_NAME / RESUME (and any other ENV below) can be
# overridden per run with `docker run -e`, which beats the baked-in ENV.
#
# ── Build: local (used by demo/live/run.sh) ──────────────────────────────────
#   docker build --target local -t terralingua-live -f demo/live/Dockerfile .
#
# ── Build: standalone (single-file handoff) ──────────────────────────────────
# Clones https://github.com/cognizant-ai-lab/terralingua over HTTPS; no key needed.
#
#   DOCKER_BUILDKIT=1 docker build --target standalone \
#       -t terralingua-live - < demo/live/Dockerfile
#
#   # Force a fresh clone on a later rebuild (the clone layer is cached otherwise):
#   DOCKER_BUILDKIT=1 docker build --target standalone \
#       --build-arg CACHE_BUST=$(date +%s) -t terralingua-live - < demo/live/Dockerfile
#
#   # A private fork over SSH: pass its URL and branch, and forward your key.
#   DOCKER_BUILDKIT=1 docker build --target standalone --ssh default \
#       --build-arg REPO_URL=git@github.com:you/your-fork.git --build-arg GIT_REF=main \
#       -t terralingua-live - < demo/live/Dockerfile
#
# ── Run (either image) ───────────────────────────────────────────────────────
#   docker run -d --name terralingua-live \
#       -p 8765:8765 \
#       --env-file demo/live/live.env \
#       -e EXP_NAME=myexp -e RESUME=false \
#       -v "$PWD/live-data:/app/data" \
#       --stop-timeout 90 \
#       terralingua-live
#
# To resume a previous run: keep the same -v data dir and EXP_NAME, pass
# -e RESUME=true. The -e EXP_NAME / -e RESUME flags are optional — omit them to
# use the defaults baked into the ENV below.

# ── base ─────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    postgresql \
    postgresql-client \
    supervisor \
    curl \
    build-essential \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# All non-secret configuration lives here as the single committed source of
# truth. Anything below can be overridden per run with `docker run -e VAR=...`.
# Secrets (FERNET_SECRET_KEY, JWT_SECRET, SESSION_SECRET, POSTGRES_PASSWORD) are
# intentionally absent — they come from --env-file at run time.

# API server. TLS is terminated upstream (ALB/nginx/Caddy), so uvicorn never
# sees the cert; FORWARDED_ALLOW_IPS lets it trust X-Forwarded-* from the proxy.
ENV API_PORT=8765 \
    API_HOST=0.0.0.0 \
    API_WORKERS=10 \
    FORWARDED_ALLOW_IPS=*

# Persistence (infra). Everything persistent lives under the bind-mounted
# /app/data: <EXP_NAME>/ (simulation), run_logs/ (service logs), pgdata/ (DB).
ENV TL_LOGS_DIR=/app/data \
    PGDATA=/app/data/pgdata \
    POSTGRES_DB=ogw_users \
    POSTGRES_USER=ogw \
    POSTGRES_HOST=127.0.0.1 \
    POSTGRES_PORT=5432

# Experiment knobs (runner). Override EXP_NAME and RESUME per run.
ENV EXP_NAME=live \
    RESUME=false \
    REMOTE_API_ENABLED=true \
    INIT_AGENTS=0 \
    MIN_AGENTS=0 \
    INIT_FOOD=500 \
    GRID_SIZE=50 \
    FOOD_ZONES=3 \
    ENV_HEARTBEAT=20 \
    MAX_TS=-1 \
    EMPTY_COUNTDOWN=-1 \
    MAX_PARALLEL_WORKERS=64 \
    SAVE_VIDEO=false

# Anthropologist (terralingua-anthropologist).
ENV ANTHRO_MODEL=claude-haiku-4-5 \
    ANTHRO_WINDOW=10 \
    ANTHRO_DETECT_INTERVAL=3 \
    ANTHRO_DISPATCH_SEVERITY=1 \
    ANTHRO_NO_AUDIT=true

# The scenario folder that holds presets, and the preset the runner starts.
# PRESET is empty by default: the live runner is configured by the ENV above.
ARG SCENARIO_DIR=scenarios/demo
ARG PRESET=
ENV TL_PRESET_ROOT=/app/${SCENARIO_DIR} \
    TERRALINGUA_PRESET=${PRESET}

EXPOSE 8765
ENTRYPOINT ["./demo/live/docker-entrypoint.sh"]

# ── local: COPY the working tree (no clone) ──────────────────────────────────
FROM base AS local
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .

# ── standalone: clone the repo at build time (no build context) ──────────────
FROM base AS standalone
ARG REPO_URL=https://github.com/cognizant-ai-lab/terralingua.git
ARG GIT_REF=main
# Bump CACHE_BUST (e.g. to a timestamp) to force a fresh clone on rebuild.
ARG CACHE_BUST=0
# The ssh mount is used only when a key is forwarded (`--ssh default`) for a
# private fork. WORKDIR created /app empty, so cloning into it is allowed;
# nothing secret is written into an image layer.
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null && \
    echo "cache-bust: ${CACHE_BUST}" >/dev/null && \
    git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" /app
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .
