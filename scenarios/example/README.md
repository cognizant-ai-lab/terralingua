# Example scenario

A small complete scenario to copy. It adds one rule set, one artifact type, and one persona to a grid world, with no change to the engine.

```
terralingua example
```

## What it shows

- **A mechanic with all four hooks** ([storms.py](storms.py)). Every `period` steps a storm forms over a random being and lasts `duration` steps. Beings within `radius` of its center lose `damage` energy per step unless a shelter stands on their cell. `on_menu` offers `build_shelter`; `on_action` carries it out; `on_step` runs the storm; `identity` gives the first being the watcher persona, and watchers get a warning one step before a storm forms.
- **An artifact type** ([artifacts.py](artifacts.py)): `shelter`, fixed, seeded by the mechanic, not creatable by agents.
- **A viewer** ([viewer/](viewer/)) in the shape the launcher starts: `python -m scenarios.example.viewer --logs logs --port 8000`. It lists the runs and shows a run's settings, latest frame and world log.
- **A preset** ([example.preset.yaml](example.preset.yaml)) that selects the scenario with `run.scenario` and sets its options in `run.scenario_options`.
- **An instructions file** ([instructions.md](instructions.md)) that tells agents how storms work, rendered with Jinja.
- **Four tests** ([tests/](tests/)): two drive the world with scripted actions and no model calls; two check the viewer's pages.

## Files

- `__init__.py`: exposes `Options` and `build(options)`. Imports `artifacts` so the type registers.
- `storms.py`: the options model and the mechanic. Its `state` dict is saved with the world checkpoint.
- `artifacts.py`: the shelter artifact.
- `example.preset.yaml`, `instructions.md`, `viewer/`, `tests/`.

## Copy it

Copy the folder, rename the package, change its imports to the new package name, and change `run.scenario` in the preset to the new module path. Run the command from the folder that holds the package, so Python can import it; `TL_PRESET_ROOT` can point at the preset folder from there. The guide is [docs/writing_a_scenario.md](../../docs/writing_a_scenario.md).
