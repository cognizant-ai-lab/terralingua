FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    postgresql \
    postgresql-client \
    supervisor \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Install the package so the terralingua, terralingua-dashboard and
# terralingua-anthropologist commands exist. Dependencies came from requirements.txt.
RUN pip install --no-cache-dir --no-deps .

# The scenario folder that holds the preset, and the preset the runner starts.
ARG SCENARIO_DIR=scenarios/demo
ARG PRESET=demo
ARG EXP_NAME=demo
ENV TL_PRESET_ROOT=/app/${SCENARIO_DIR} \
    TERRALINGUA_PRESET=${PRESET} \
    TERRALINGUA_EXP_NAME=${EXP_NAME}

# Ephemeral Postgres data dir — local demo state lives inside the container
# and is discarded when the container is removed.
RUN mkdir -p /app/logs /app/data /app/pgdata && chown -R postgres:postgres /app/pgdata

ENV DEMO_MODE=true \
    API_PORT=8765 \
    API_HOST=0.0.0.0 \
    API_WORKERS=4 \
    FORWARDED_ALLOW_IPS=127.0.0.1 \
    PGDATA=/app/pgdata \
    POSTGRES_DB=ogw_users \
    POSTGRES_USER=ogw \
    POSTGRES_HOST=127.0.0.1 \
    POSTGRES_PORT=5432

EXPOSE 8765

ENTRYPOINT ["./demo/local/docker-entrypoint.sh"]
