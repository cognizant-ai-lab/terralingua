# Paper experiments

One preset per experimental condition of the TerraLingua paper. Run one from the repo root:

```
terralingua paper_core
```

`paper_core` is the baseline. The other seven change one thing each: `paper_abundant` (long history and food everywhere), `paper_artifact_cost`, `paper_creative`, `paper_inert_artifacts`, `paper_long_memory`, `paper_no_motivation`, `paper_no_personality`.

## How the presets map to the original scripts

The original shell scripts used an older command line. The presets carry the same values under the current field names:

- `exogenous_motivation` is `agent.scenario_specific_instructions`. The builtin texts `base`, `creative`, and `none` are unchanged.
- `dead_agent_food single` is `env.drop_food_on_death: true`.
- `food_zones null` keeps its meaning: food is placed uniformly.
- `save_root` is gone. Logs go to `logs/` under the working directory.

Four defaults changed after the paper. The presets set the paper values: `agent.internal_memory_size: 150`, `env.food_spawn_rate: 1`, `env.dynamic_grid_scaling: false`, and `run.env_heartbeat: -1`. The last one makes each step wait for every agent's decision, as the original runner did; the current default gives agents 30 seconds and then makes them stay.

Mechanics added after the paper keep their current defaults, because the paper had no value for them: two-parent spawning, the genome mutation rate, the message and artifact token limits, fixed artifacts, and the reproduction population cap. Each being's lifespan is now drawn within 20% of `agent_lifespan`, so these runs give lifespans between 80 and 120 steps; the paper's runner used the exact value, and no setting turns the variation off. The `obs_style` setting of the old code no longer exists.

The model name `DeepSeek-R1-32` is a local model served on the ports in `run.ports`. Use `--model` to run with another model.
