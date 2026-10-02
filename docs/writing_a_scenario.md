# Writing a scenario

A scenario adds rules, artifact types, and personas to a world without editing the engine. It is one Python package plus a preset. The engine calls the scenario's code at fixed moments of each step. This page explains those moments and what the code may read and write.

The folder `scenarios/example/` is a small complete scenario. Copy it to start.

## The package

A scenario package exposes two names:

- `Options`: a pydantic model. The preset's `run.scenario_options` is validated against it. Use `extra="forbid"` so a typo in the preset fails at start.
- `build(options)`: returns a list of mechanics.

```python
from scenarios.example import artifacts  # registers the artifact types on import
from scenarios.example.storms import Storms, StormsOptions

Options = StormsOptions


def build(options: StormsOptions):
    return [Storms(options)]
```

The preset selects the package by module path:

```yaml
config:
  run:
    scenario: scenarios.example
    scenario_options:
      period: 5
```

The runner imports the module when it starts, before it reads any external server configuration. Anything registered at import time, such as artifact types, is in place from then on. The runner then calls `build` when it creates the world, and again on resume.

The package can live under `scenarios/` in this repository, or in its own repository that installs `terralingua`. The module path must be importable: run the command from the folder that holds the package (the runner puts the working directory on the import path), or install the package with pip.

## The mechanic

A mechanic is a subclass of `terralingua.environment.mechanic.Mechanic`. It has three hooks and one lookup. Every one does nothing by default. A subclass that defines `__init__` must call `super().__init__()`.

| Hook | When the world calls it | What it may do |
|---|---|---|
| `on_menu(env, tag, menu) -> menu` | Whenever the world builds a being's action menu: at the end of each step N for step N+1, at restart, and when a being is added. The menu already has affordances, removals, and exclusions applied. | Drop, add, or reword menu entries. |
| `on_action(env, tag, action, params) -> str or None` | Right before the world executes one being's action in step N, after the action passed the menu and parameter checks. Earlier beings' actions of the same step are already applied. | Return `None` to let the world run the action. Return a string to take the action over: the world skips its own handling, shows the string to the being under "Action outcome", and logs a `MECHANIC_ACTION` event. Use it to refuse an action with a reason, or to carry out an action the mechanic added to the menu. |
| `on_step(env, infos)` | Once per step, after all actions and artifact updates, before energy drain and deaths. | Change state, energy, notes, artifacts. Changes appear in the observations sent at the end of the step. |
| `identity(env, tag) -> dict or None` | Once when the runner creates a new being, before the world adds it. Not called on resume. | Return `{"name": ..., "persona": ...}`. Both keys are optional. The name replaces the generated one. The persona is appended to the being's personality in the system prompt. |

Mechanics run in the order `build` returns them. The menu one mechanic returns is the input of the next. The first string from `on_action` wins. The first identity wins.

`on_action` cannot stop an external tool action (`<server>_<tool>`). The runner sends those to the MCP server before the world step.

### A menu entry

```python
menu["build_shelter"] = {
    "description": "Build a shelter on your cell. It costs 5 energy.",
    "params": {},
}
```

A parameter is either a text description, or a dict with `description` and `choices`:

```python
menu["give"]["params"]["target"] = {"description": "A being next to you.", "choices": ["Ada", "Bo"]}
```

An action that exists only in the mechanic must be handled in `on_action`. The world knows nothing about it.

### What a mechanic may read

| Name | Content |
|---|---|
| `env.step_count` | The current step, starting at 0. |
| `env.agent_registry` | The tags of the living beings. |
| `env.agent_pos[tag]` | A being's position: `(row, col)` on a grid, a node id on a graph. |
| `env.agent_names[tag]`, `env.name_to_tag[name]` | Names are what beings see. Tags are what the engine uses. |
| `env.agent_energy[tag]` | Energy. |
| `env.agent_inventories[tag]` | Names of the artifacts a being carries. |
| `env.artifacts[name]` | The artifact objects. `env.pos_artifacts[pose]` holds the names on a cell or node. `env.artifact_location[name]` is `("map", pose)` or `("inv", tag)`. |
| `env.agents_within(tag, r)` | Tags of the other beings within distance `r`, sorted. |
| `env.distance(a, b)` | Distance between two positions in the world's own measure: Chebyshev on a torus grid, hops on a graph. |
| `env.transfers` | This step's successful `give`, `take`, and `give_artifact` as `(giver, receiver, kind)`. Rebuilt every step. |
| `env.deaths` | Records of the previous step's deaths: tag, name, position, reason, artifacts, step. |
| `env.rng` | A numpy generator seeded by `run.seed`. Use it for every random choice, so a run repeats. |
| `env.logger` | The world log. `None` in some tests. |
| `env.log_path` | The run folder, `logs/<exp_name>/`. A mechanic may write its own files there. |

### What a mechanic may write

| Call | Effect |
|---|---|
| `env.note(tag, key, text)` | Adds a line under `key` to the being's next observation. Several notes under one key are joined with line breaks. |
| `env.kill(tag, reason)` | Queues a death. The world kills the being in this step's death phase. The reason appears in the `AGENT_DIED` event. |
| `env.agent_energy[tag] -= amount` | Energy changes take effect at once. |
| `env.seed_artifact(pose, art_type, name, payload, lifespan, movable=True, to_inventory=None, **params)` | Places an artifact without an energy check. Returns `(message, final_name)`. The name gets a suffix on collision. `params` go to the artifact class. |
| `env.add_artifact(pose, art_type, name, payload, creator, lifespan, movable=True, to_inventory=None, **params)` | The same, charged to `creator` at the world's artifact cost. |
| `env.no_appetite` | A set of tags. A being in the set leaves the food on its cell untouched. The world checks it right after each move, so a tag added in step N takes effect from step N+1. |
| `env.logger.log(time=env.step_count, event_type="MY_EVENT", **fields)` | Writes one line to the world log. Use an upper-case string as the event name. Check `env.logger` first. |

### State and checkpoints

`self.state` is a dict. The world saves it under the mechanic's name in every checkpoint and restores it in place on resume. Keep only values that JSON and pickle both accept: numbers, strings, lists, dicts. Store a grid position as a list, and turn it back into a tuple when you read it. The mechanic's name is the class name in lower case unless the class sets `name`.

## Your own outputs and viewers

The core dashboard shows the world: positions, energy, inventory, artifacts, births and deaths. It knows nothing about a scenario's state. A scenario that wants to show its own state ships its own viewer, as a separate process that reads the run folder. The mechanic writes what that viewer needs: from `on_step`, append one JSON line per step to a file under `env.log_path`, with the positions, the energy and the mechanic's own state, and log the scenario's events with `env.logger.log`. The viewer then works on a finished run and on a running one.

## Artifact types

An artifact type is a subclass of `terralingua.environment.artifact.Artifact` with the `register_artifact_type` decorator:

```python
@register_artifact_type("shelter")
class ShelterArtifact(Artifact):
    description = "A shelter. Beings on its cell are safe from storms."
```

- `description` is what beings read about the type.
- `creatable` is `False` by default, so beings cannot make it with `create_artifact`. Set it to `True` to let them.
- The class must define `actions` (a property, a dict of the actions beings can take on it), `interact`, `passive_effect` (the text a being sees when it stands next to it), and `verify_payload`.
- Extra constructor arguments, such as a radius, come from `**params` of `seed_artifact`. Override `serialize` and `deserialize` to save and restore them with the world.
- Import the module that defines the type from the package's `__init__.py`, so the type registers when the scenario loads.

A preset can also seed artifacts from JSON files with `env.init_artifacts_path`. Each entry names the `art_type` and may carry a `params` dict for the extra arguments.

## A viewer for the runs

A scenario may ship tools that read a run folder: a viewer, an anthropologist. Put each one in a subpackage named `viewer` or `anthropologist`, runnable as

```
python -m <scenario>.viewer --logs <folder> --port <n>
```

`--logs` is the folder that holds the runs. `--port` is the port of the page the tool serves. `describe --preset` reports the tools a scenario ships under `scenario.tools`, so the launcher can show a button that starts them. The example scenario has a small viewer in this shape.

## Personas and names

`identity(env, tag)` runs once per new being, at the initial population and at every respawn. It sees the world, so it can count what it has already handed out and keep the record in `self.state`. Return a `persona` to append text to the being's personality, a `name` to replace the generated one, or both. The being's checkpoint keeps them, so resume does not call the hook again.

A preset can also hand out personas from a file with `agent.personas_path`, without any code. The file holds a JSON list. An entry is the persona text, or an object with `persona`, an optional `name` and an optional `count` (default 1). The runner gives the entries to new beings in creation order, at the initial population and at respawns, until the list is used up. Children born from a `spawn` action take none. A persona returned by a mechanic wins over the file. A name applies only to an entry with count 1, and a name already in use is not applied.

```json
[
  {"persona": "You are a healer. You look for sick beings and treat them.", "name": "Ada"},
  {"persona": "You are a farmer. You stay near the fields.", "count": 3},
  "You doubt everything you hear."
]
```

## The beings' instructions

`agent.scenario_specific_instructions` is a builtin name (`base`, `creative`, `survival`, `none`) or a path to a Markdown file. A path in a preset is relative to the preset's folder. The file is rendered with Jinja. Two variables are set: `finite_energy` and `finite_lifespan`. A file can include partials from its own folder, and from any path under `<working directory>/scenarios/`, for example `{% include "_prompts/shared.md" %}`.

Write facts about how the world works. Do not tell beings what to do with them. Action texts belong in the menu, not here.

## Roles and affordances

Every world supports persistent roles and location permissions. `env.roles_hocon_path` names a HOCON file of roles with a charter, a capacity, a mandate, and lifecycle rules. `env.affordances_file_path` names a JSON file that attaches actions to places or roles: on a grid the keys are `"row,col"` cells, on a graph they are node ids, and in roles mode they are role names. A permission can add an action with an effect the world knows how to run, or remove a global action such as `spawn`. A mechanic can do most of this in `on_menu` and `on_action` as well. Use the files when the rule is static and per place, and a mechanic when it depends on state.

## The preset

A preset is a file named `preset.yaml` or `<name>.preset.yaml`:

```yaml
name: example
description: One line shown by --list.
config:
  run: {...}
  agent: {...}
  env: {...}
```

Path values (`scenario_specific_instructions`, `roles_hocon_path`, `affordances_file_path`, `init_artifacts_path`, `external_servers_config`, graph files) are relative to the preset's folder, so a scenario folder is self-contained.

`terralingua` finds presets under the working directory, or under `TL_PRESET_ROOT`. Run a scenario from the folder that holds its package, here the repository root:

```bash
terralingua example
TL_PRESET_ROOT=scenarios/example terralingua example --max_ts 10
```

In a scenario's own repository the layout is the same: the package folder and the preset at the top, and the command runs from there.

Options can be overridden on the command line as JSON: `--scenario_options '{"period": 3}'`.

Every new engine setting a scenario needs is a change to the engine, not to the scenario. If a scenario needs one, it belongs in a discussion, not in a copy of the engine.

## When an option applies

Some options matter only when another option turns a rule on. The package may say so with an `APPLICABILITY` table, in the shape the engine uses for its own settings: option name to (condition, reason). Conditions use the engine's helpers over option names. Nested options use dotted names.

```python
from terralingua.config.dependencies import when

APPLICABILITY = {
    "storm_damage": (when("storms", const=True), "Requires storms."),
    "shelter.capacity": (when("shelter", type="object"), "Requires a shelter."),
}
```

The engine reads the table when it composes a run, so it imports the package then. A run that sets an inactive option to a value other than its default gets a warning at start. `python -m terralingua.config evaluate --preset <name>` lists each declared option under `run.scenario_options.<name>` with its state, and `describe --preset <name>` lists every option with its type, default, description and rule. Conditions are checked against the validated options, so defaults count. A table that names an unknown option, or a condition that is not valid JSON Schema, is an error. Options the package rejects fail the composition with the package's own message. When the package cannot be imported from the working directory, `evaluate` says so and leaves the check to the start of the run, and `describe` reports the import error.

## Tests

Drive the world directly, with scripted actions and no model calls. The pattern:

```python
env = OpenGridWorld(grid_size=8, init_food=0, food_mechanism=False, log_path=tmp_path, headless=True, ...)
env.attach(Storms(StormsOptions(period=2)))
env.add_agent("a0", "Ada", "text", position=(3, 3))
env.restart_env(seed=3, agent_poses={"a0": (3, 3)})
infos = env.step({"a0": {"action": "move", "params": {"direction": "stay"}}})[-1]
```

`infos[tag]` holds what the being will read: notes under their keys, and `"Action outcome"` for a string returned by `on_action`. `env.agent_avail_actions[tag]` is the menu for the next step. The world log is a JSONL file at `env.logger.save_path`. Keep the tests few: one per rule that matters.

A test that composes a preset by name needs the working directory to hold it. The repository's `conftest.py` sets the working directory to the repository root for every test.
