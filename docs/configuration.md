# Inspect configuration without starting a run

Run these commands from the repository root.
They read configuration and print JSON.
They do not start agents, external services, or the dashboard.

## Commands

Describe all fields, defaults, choices, and dependencies:

```sh
python -m terralingua.config describe
```

Also describe the options of a scenario, either through a preset that selects it or by module name:

```sh
python -m terralingua.config describe --preset example
python -m terralingua.config describe --scenario scenarios.example
```

Evaluate an installed preset:

```sh
python -m terralingua.config evaluate --preset example
```

Apply flat overrides:

```sh
python -m terralingua.config evaluate --preset example --overrides '{"reproduction_cost": 10, "food_mechanism": false}'
```

Evaluate a nested JSON file:

```sh
python -m terralingua.config evaluate --config config.json
```

A nested file can contain only the fields being changed:

```json
{
  "agent": {
    "max_history": 10,
    "internal_memory_size": 500
  },
  "env": {
    "food_mechanism": false,
    "reproduction_cost": 10
  }
}
```

Use `--config -` or `--overrides -` to read standard input.
Only one argument can read standard input.
`--config` otherwise expects a file path.
`--overrides` otherwise expects a JSON object.

Successful evaluation exits with status 0.
Invalid settings exit with status 2.
Argument syntax errors also use status 2 and show usage instructions.

List the presets, with their descriptions and locations:

```sh
python -m terralingua.config presets
```

List the artifact types a run can seed, with the parameters each type adds. With a preset, the types its scenario registers appear too:

```sh
python -m terralingua.config artifact-types --preset example
```

Print the installed package version:

```sh
python -m terralingua.config version
```

Every command prints one JSON object. A failure prints `{"valid": false, "diagnostics": [...]}` and exits with status 2.

## Composition

Values apply in this order:

1. Model defaults.
2. Named preset.
3. Nested `config` input.
4. Flat `overrides` input.

Bare names and canonical paths are accepted in flat overrides.
For example, `max_ts` and `run.max_ts` identify the same field.
Graph settings also accept `graph.topology` for `env.graph.topology`.

In a graph world a move may cost energy per edge. Set `graph.move_cost_attr` to the name of an edge attribute in the graph file (for example `energy_cost`); a move along an edge then costs that many energy units, and an edge without the attribute costs `graph.default_move_cost` (1). Staying costs nothing extra, the per-step upkeep is unchanged, the move menu lists the cost of every exit, a being without enough energy keeps its place and reads why, and every charge is a `MOVE_COST` event in the world log. Leave `move_cost_attr` unset for the previous behaviour.

The energy every being loses per step is `energy_upkeep` (null: 1 with `food_mechanism`, 0 without). Set `food_mechanism: false` with `energy_upkeep: 1` for a world where energy drains every step but no food appears, for example when a scenario supplies energy through its own artifacts; `energy_death` then defaults to true and the system prompt describes the drain without mentioning food.

Supply each setting once within an override layer.
Aliases and parent/child paths cannot overlap within that layer.
Repeated CLI flags and duplicate JSON or YAML keys are rejected.
Different layers can override the same setting normally.

Unknown fields fail validation.
Dictionary settings allow their own keys.
For example, `run.scenario_options.seed` targets the selected runner's options.
That runner validates the option's meaning at startup.

Relative paths inside presets resolve against the preset directory.
Builtin instruction names remain unchanged.
Paths in explicit config and overrides use the current working directory.

Social graphs do not support natural food.
They default `food_mechanism` to false and resolve an explicit true value to false. The evaluation reports it as a normalization.
Their energy fees, rewards, transfers, and reproduction still work.

Set `internal_memory_size=0` to disable private memory.
Positive values set its exact token budget.
History length remains independent.

Negative initial energy gives founders and free newborns unlimited energy.
An agent with unlimited energy receives no own-energy status, survival rules, or action-cost prose.
Transfer descriptions and a paid newborn's finite energy remain visible.
Paid births still start with the reproduction cost plus any parent-funded gift.

Set `agent_lifespan=-1` for unlimited life.
Agents then receive no remaining-life value or age-expiry instructions.
Simulation steps, history ordering, artifact lifetimes, and scenario clocks remain visible.
Finite life continues to count down normally.

Use JSON `null` for nullable values.
Do not substitute a string such as `"null"` unless that field supports it.

## Scenario options

A scenario package may declare which of its options apply only when another option turns a rule on; see [writing_a_scenario.md](writing_a_scenario.md). `evaluate` then lists those options under `run.scenario_options.<name>` in `fields`, `active_values` and `inactive_values`, and warns about inactive options the run sets.

`describe --preset <name>` or `describe --scenario <module>` adds a `scenario` key: the module name, the JSON Schema of its options model, and `fields`, keyed by `run.scenario_options.<name>`, nested options with dotted names. Each field has `type`, `schema`, `default`, `required`, `description`, `active_when`, `inactive_reason`, `depends_on` and `affects`, like a core field. A preset without a scenario gives `null`. Plain `describe` has no `scenario` key. A scenario that cannot be imported from the working directory, or an options model whose types have no JSON Schema, gives an error report with exit code 2.

## Description output

`describe` includes:

| Key | Meaning |
| --- | --- |
| `format_version` | Version of the response structure. |
| `schema_hash` | Fingerprint of the description contract. |
| `json_schema` | Pydantic's configuration schema. |
| `fields` | Field descriptions keyed by canonical paths. |
| `groups` | Field groups for navigation. |
| `constraints` | Explanations of cross-field constraints. |
| `derived_dependencies` | Inputs used by selected derived values. |

Each field includes its schema, default, aliases, group, and notes.
The food switch also declares its social-world default in default_when.
`active_when` uses JSON Schema 2020-12.
Its context is a dictionary keyed by canonical field paths.
Dictionary-valued parameters remain whole values in that context.

`depends_on` identifies applicability inputs.
`affects` lists the reverse applicability links.
These lists do not claim to cover every runtime data dependency.

Use `evaluate` for authoritative validation.
Constraint descriptions and JSON Schema do not replace Pydantic's validators.

## Evaluation output

| Key | Meaning |
| --- | --- |
| `valid` | Whether configuration validation passed. |
| `input` | Submitted preset, config, and override layers. |
| `requested` | Merged input before defaults and model normalization. |
| `resolved` | The validated nested configuration, or null on failure. |
| `fields` | Applicability state and reason for each field. |
| `active_values` | Active values keyed by canonical path. |
| `inactive_values` | Stored values that do not apply in this configuration. |
| `normalizations` | Reported model changes. |
| `derived` | Selected runtime meanings, such as effective energy-death behavior. |
| `diagnostics` | Errors, warnings, and deferred validation notices. |

Values remain stored when inactive.
Do not reset them when disabling a control.
They may become active after another setting changes.
Normal run composition logs warnings for changed inactive values and normalizations.
Spatial food zones remain applicable to manual food injection when natural food is disabled.

Ordinary type coercion is not necessarily listed as a normalization.
Compare `input`, `requested`, and `resolved` when exact input provenance matters.
If input fails before composition, `requested` can be null.
Inputs that cannot be represented as valid JSON have `input=null`.

Derived newborn energy is a number, `"unlimited"`, or null when reproduction is disabled.
Other runtime transformations are not all materialized in `derived`.
The resolved object is a validated model, not a complete runtime manifest.

## Python API

```python
from terralingua.config.evaluation import evaluate
from terralingua.config.inspection import describe

contract = describe()
result = evaluate(
    preset="example",
    config={"agent": {"max_history": 10}},
    overrides={"reproduction_cost": 10},
)
```

The evaluation API accepts JSON-compatible values.
It returns diagnostics rather than starting a simulation.

## Validation limits

Offline evaluation does not contact model providers or external servers.
It does not import custom runner classes or custom topology factories.
It imports the package named by `run.scenario`, with the working directory on the import path, to read the scenario's options model and applicability table. The package's import-time code runs.
Server contents, permissions, and connectivity still need startup checks.
Scenario runners validate their own option dictionaries.
Referenced files can change between evaluation and startup.

The schema hash identifies the description contract.
It does not hash prompt files, external configuration, or checkpoint state.


Artifact creation uses `env.artifact_creation_cost`: `-1` disables creation,
`0` makes it free, and positive values charge that much energy.
This setting does not disable access to existing or seeded artifacts.
