# TerraLingua demos

Two containerized flavours of the same TerraLingua stack. Both run Redis, the API server, the simulation runner, and the anthropologist in one container (base image `python:3.12-slim`) and serve the dashboard on `http://localhost:8765`. Setup and launch instructions are in each subdirectory's README.

- [`local/`](local/) — single-user, ephemeral. Auto-seeds a demo user, auto-opens the browser, runs a short scenario. Best for a quick showcase.
- [`live/`](live/) — multi-user, persistent. Users sign up; data lives on bind-mounted host directories so restarts don't wipe state. Mirrors `terralingua.evolution.ml`.

| | Local | Live |
|---|---|---|
| Account / auth | Pre-seeded `demo@demo.com`, `/auto-login` | Sign up at `/register` |
| Persistence | None (DB recreated on each launch) | Bind-mounted host dirs (Postgres + logs) |
| Experiment params | The `demo` preset (`scenarios/demo/preset.yaml`) | Configurable via Dockerfile `ENV` / `docker run -e` |
| Browser auto-open | Yes | No |
| Recommended for | Showcasing, smoke tests | Real deployments |
