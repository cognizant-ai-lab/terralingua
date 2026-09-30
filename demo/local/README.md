# TerraLingua local demo

Runs the full TerraLingua stack (Redis, API server, runner, anthropologist) in a single Docker container — no account setup, no API key entry in the UI. For a persistent multi-user deployment, see [`../live/`](../live/).

## Prerequisites

- Docker installed and running
- An Anthropic API key (`sk-ant-...`)

## Launch

```bash
ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh
```

The script builds the image (first run takes a few minutes), starts the container, waits until everything is ready, and opens the dashboard at `http://localhost:8765` already logged in as the demo user. By default it runs the empty grid world: food spawns over time and agents arrive via the dashboard.

To seed from a Neuro-SAN agent-network HOCON instead, pass `--hocon` (uses the bundled `demo/local/neuro_san_network.hocon`), or set `NEURO_SAN_HOCON` to another file (path relative to the repo root or absolute):

```bash
ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh --hocon
NEURO_SAN_HOCON=/path/to/network.hocon ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh
```

## Stop

```bash
docker stop terralingua-demo
```

## Logs

Logs are bind-mounted to `logs_demo/local/`. Stream the mixed container log with `docker logs -f terralingua-demo`, or a single service:

```bash
tail -f logs_demo/local/runner.log          # also: anthropologist.log, api.log, redis.log
```
