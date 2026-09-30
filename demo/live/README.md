# TerraLingua live demo

Containerized version of the long-running, multi-user deployment (the same stack that powers `terralingua.evolution.ml`). Differs from [`../local/`](../local/):

- **Multi-user**: users sign up at `/register` and sign in at `/login`. No demo user, no `/auto-login`, no browser auto-open.
- **Persistent state** under one bind-mounted host directory (`HOST_DATA_PATH`, default `./logs_demo/live/`): `<EXP_NAME>/` (checkpoints + agent logs + field notes), `run_logs/` (service logs), `pgdata/` (Postgres). Restarting does **not** wipe users or progress.
- **Config baked into the image**: non-secret experiment parameters live in the [`Dockerfile`](Dockerfile) `ENV`; secrets come from an uncommitted `live.env` at run time.

No platform Anthropic API key is required — each registered user provides their own from the dashboard. Optionally set `ANTHROPIC_API_KEY` in `live.env` as a fallback for server-side LLM calls with no per-user context.

## First-time setup (secrets)

```bash
cp demo/live/live.env.example demo/live/live.env
# Fill in FERNET_SECRET_KEY, JWT_SECRET, SESSION_SECRET, POSTGRES_PASSWORD.
# Generation commands are in the file. The entrypoint refuses to boot if any is
# empty. Generate them ONCE and reuse across restarts — changing them invalidates
# stored API keys and sessions.
```

## Run

The Dockerfile is multi-stage with two targets that produce the same `terralingua-live` image:

**Local** (builds the `local` target from the working tree, bind-mounts `./logs_demo/live/` to `/app/data`, waits for `/api/stats`):

```bash
./demo/live/run.sh                # use the EXP_NAME / RESUME baked into the image
./demo/live/run.sh --resume       # resume from the latest checkpoint
./demo/live/run.sh --exp myexp    # run/resume experiment "myexp"
```

**Standalone** (the `standalone` target clones the public repository itself; for a private fork pass `--build-arg REPO_URL=... --build-arg GIT_REF=...` and `--ssh default`):

```bash
DOCKER_BUILDKIT=1 docker build --target standalone \
    -t terralingua-live - < demo/live/Dockerfile

docker run -d --name terralingua-live -p 8765:8765 \
    --env-file demo/live/live.env \
    -e EXP_NAME=myexp -e RESUME=false \
    -v "$PWD/logs_demo/live:/app/data" --stop-timeout 90 \
    terralingua-live
```

`-e EXP_NAME` / `-e RESUME` are optional (defaults are baked in). Add `--build-arg CACHE_BUST=$(date +%s)` to force a fresh clone on rebuild.

## Stop

```bash
docker stop terralingua-live   # user DB and logs are preserved; re-running resumes
```

## Configuration

Non-secret defaults are the `ENV` lines in the [`Dockerfile`](Dockerfile); override any for a single run with `docker run -e VAR=value` (no rebuild). Notable ones:

| Variable | Meaning |
|---|---|
| `EXP_NAME` | Subdir of `HOST_DATA_PATH` where the sim writes (`HOST_DATA_PATH/<EXP_NAME>/`) |
| `RESUME` | `true` to continue a previous run with the same `EXP_NAME`; `false` for fresh |
| `INIT_AGENTS` / `MIN_AGENTS` / `INIT_FOOD` / `GRID_SIZE` / `FOOD_ZONES` | Initial world conditions |
| `MAX_TS` | `-1` for unlimited; otherwise stop after N ticks |
| `ENV_HEARTBEAT` | Heartbeat tick rate (lower = faster) |
| `MAX_PARALLEL_WORKERS` | Concurrency cap on the runner's LLM fanout |
| `ANTHRO_*` | Anthropologist knobs (model, detection interval, severity) |
| `API_WORKERS` | uvicorn worker process count |
| `FORWARDED_ALLOW_IPS` | IPs/CIDRs trusted to set `X-Forwarded-*` (needed for rate-limit IP detection behind a proxy; defaults to `*`) |

## Logs

```bash
docker logs -f terralingua-live                       # mixed stream
tail -f logs_demo/live/run_logs/runner.log            # per-service: api, runner,
                                                      # anthropologist, postgres, redis, supervisord
```

## Behind a load balancer

If an ALB / nginx / Caddy terminates TLS: leave `COOKIE_SECURE` unset (it defaults to `true`, correct over HTTPS), and narrow `FORWARDED_ALLOW_IPS` to the proxy CIDR with `-e FORWARDED_ALLOW_IPS=...` if you don't want the default `*`.
