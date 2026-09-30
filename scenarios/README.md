# Scenarios

A scenario adds rules, artifact types, and personas to a world without editing the engine. Each folder holds only what is specific to it: a preset, and where needed a Python package, an instructions file, and tests. Shared code lives in `terralingua/`, the engine.

| Folder | What it is |
|---|---|
| [example](example/) | A small complete scenario to copy: a storms mechanic, a shelter artifact, a persona, two tests. |
| [demo](demo/) | The preset behind the dashboard demo. No code. |
| [paper](paper/) | One preset per experimental condition of the TerraLingua paper. |

Run a preset from the repository root:

```
terralingua example
```

How to write one: [docs/writing_a_scenario.md](../docs/writing_a_scenario.md). How external tool servers plug in: [docs/external_tools.md](../docs/external_tools.md).
