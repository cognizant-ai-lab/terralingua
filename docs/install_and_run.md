# Install and run TerraLingua

This page covers the install, a first simulation, the dashboard, the live anthropologist, the analysis scripts, and the Docker demo.

## Requirements

- Python 3.10 or newer.
- An LLM: an Anthropic API key by default. OpenAI and Bedrock keys work through litellm. Local vLLM servers work through `run.ports`.
- Redis and PostgreSQL, only for the dashboard.
- Docker, only for the demo container.

## Install

From a checkout of the repository:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                          # the package and the three commands
pip install -e ".[analysis]"              # optional: the libraries the notebooks use
python -m spacy download en_core_web_sm   # once: the anthropologist's artifact analysis needs it
```

The install gives three commands:

| Command | What it does |
|---|---|
| `terralingua` | Runs a simulation. `python main.py` does the same. |
| `terralingua-dashboard` | Serves the dashboard and the remote-agent API on port 8765. `python api_server.py` does the same. |
| `terralingua-anthropologist` | Runs the live analysis next to a run. `python anthropologist_server.py` does the same. |

Developers who run the full test suite also install `requirements.txt`.

## Keys and secrets

The commands read a `.env` file from the working directory, or from a parent folder. Put the LLM key there:

```
ANTHROPIC_API_KEY=your_key_here
```

The dashboard also needs a database and three secrets. Generate the secrets once and keep them. A new `FERNET_SECRET_KEY` invalidates every stored user API key.

```bash
echo "DATABASE_URL=postgresql://user:password@localhost:5432/terralingua"
python -c "from cryptography.fernet import Fernet; print('FERNET_SECRET_KEY=' + Fernet.generate_key().decode())"
python -c "import secrets; print('JWT_SECRET=' + secrets.token_hex(32))"
python -c "import secrets; print('SESSION_SECRET=' + secrets.token_hex(32))"
```

## Run a simulation

```bash
terralingua --list                                  # the presets found under the working directory
terralingua grid_baseline                           # a built-in preset
terralingua grid_baseline --max_ts 50 --init_agents 5   # with overrides
terralingua --help                                  # every setting and its default
terralingua grid_baseline --resume                  # continue from the last checkpoint
```

A run composes its settings in three layers: model defaults, then the preset, then the command line. Any `--field VALUE`, `--section.field VALUE`, or `--no-flag` token is an override.

Presets are files named `preset.yaml` or `<name>.preset.yaml`. The command finds them under the working directory, so run it from the repository root. `TL_PRESET_ROOT` points it at another folder. `terralingua --list PATH` lists the presets under a path. A preset that names a scenario package needs that package importable from the working directory.

A run writes to `logs/<exp_name>/` under the working directory: the agent logs, the world log, `params.json`, frames, and checkpoints. `TL_LOGS_DIR` moves that folder. Set it in the shell, not in `.env`.

### Seed artifacts at start

`env.init_artifacts_path` (or `--init_artifacts_path ./my_artifacts/`) names a folder of JSON files. Each file holds one artifact or a list of them. Files are read in alphabetical order.

```json
{
  "name": "founding_charter",
  "art_type": "text",
  "payload": "We the beings of this world...",
  "pose": [25, 25],
  "lifespan": -1
}
```

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Unique name. A collision adds `_1`, `_2`, and so on. |
| `art_type` | Yes | `text`, or a type registered by the scenario. |
| `payload` | Yes | The content. Text is capped at `env.max_artifact_tokens`. |
| `pose` | Yes | `[row, col]` on a grid, a node id on a graph. |
| `lifespan` | No | Steps until the artifact expires. `-1` (default) means never. |
| `params` | No | Extra constructor arguments of a scenario artifact type. |

Seeded artifacts have the creator `system` and skip the energy check.

Inspect a configuration without running anything:

```bash
python -m terralingua.config describe
python -m terralingua.config evaluate --preset grid_baseline
```

See [configuration.md](configuration.md) for these commands.

## The dashboard

The runner and the dashboard are two processes. They talk through Redis.

1. Start Redis and PostgreSQL. Put `DATABASE_URL` and the three secrets in `.env`.
2. Start the dashboard: `terralingua-dashboard --dev`. The `--dev` flag allows plain HTTP cookies on localhost.
3. Start a run with the remote API on: `terralingua grid_baseline --remote_api_enabled`.
4. Open `http://localhost:8765/dashboard`.

`run_suite.sh` starts Redis, the dashboard, a run, and the anthropologist in one tmux session. It needs `tmux` and `redis-server`.

### A public server

uvicorn serves HTTPS, the static files, and the WebSockets itself; no reverse proxy is needed. Pass the certificate files and a worker count:

```bash
terralingua-dashboard --workers $(nproc) \
    --ssl-certfile /etc/letsencrypt/live/example.com/fullchain.pem \
    --ssl-keyfile  /etc/letsencrypt/live/example.com/privkey.pem
```

`SSL_CERTFILE`, `SSL_KEYFILE`, and `API_WORKERS` are the matching variables. More than one worker turns on multi-process mode; the workers share state through Redis. Behind a load balancer, set `FORWARDED_ALLOW_IPS` to the proxy's address so that rate limits key on the real client address. WebSocket connections upgrade to `wss://` on their own.

## The live anthropologist

The anthropologist reads a run's logs while the run is going and publishes its notes to the dashboard through Redis.

```bash
terralingua-anthropologist --exp_name <exp_name>
```

Run it from the same working directory as the run, so it finds `logs/<exp_name>/`. It writes `logs/<exp_name>/field_notes.jsonl`.

## Analysis after a run

The numbered scripts under `analysis_scripts/` run the anthropologist's steps over finished runs. Set `EXPERIMENTS_NAMES` at the top of a script and run it from the repository root:

```bash
python analysis_scripts/001_llm_agent_analyser.py
```

The notebooks under `analysis_scripts/notebooks/` show each step. They need the `analysis` extra. Start Jupyter from the repository root and open a notebook from that folder.

## The demo

The Docker images under `demo/` hold everything: PostgreSQL, Redis, the dashboard, a run, and the anthropologist.

Local demo, one user, nothing kept after the container stops:

```bash
ANTHROPIC_API_KEY=sk-ant-... ./demo/local/run.sh
```

It builds the image, starts the container, and prints `http://localhost:8765/auto-login`. Stop it with `docker stop terralingua-demo`.

The image runs the `demo` preset from `scenarios/demo/`. To bake in another scenario folder and preset:

```bash
SCENARIO_DIR=scenarios/my_scenario PRESET=my_preset ./demo/local/run.sh
```

Live demo, many users, data kept on the host: copy `demo/live/live.env.example` to `demo/live/live.env`, fill in the secrets, and run `./demo/live/run.sh`. See [demo/README.md](../demo/README.md) for the two flavours side by side.

## Write your own scenario

See [writing_a_scenario.md](writing_a_scenario.md). The folder `scenarios/example/` is a small complete scenario to copy.
