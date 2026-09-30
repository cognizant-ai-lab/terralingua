# Demo scenario

The preset behind the dashboard demo. It has no code: an empty 50x50 grid that runs until stopped, with the remote agent API on, so people add beings from the dashboard.

```
terralingua demo --remote_api_enabled     # from the repo root; needs Redis and terralingua-dashboard
```

The local Docker image under `demo/local/` builds with this folder as the preset root (`TL_PRESET_ROOT`) and runs `terralingua demo`; the live image under `demo/live/` takes its settings from the ENV lines of its Dockerfile. Pass `--build-arg SCENARIO_DIR=... --build-arg PRESET=...` to build an image for another scenario folder and preset.
