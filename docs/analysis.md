# Analysis

## What a run writes

A run writes `logs/<exp_name>/` under the working directory:

- `agent_logs/`: one JSONL file per being with every observation, decision, and message, plus its genome.
- `open_gridworld.log`, `open_graph_world.log`, or `open_social_graph_world.log`, depending on the world type: the world log, one JSON line per event (moves, births, deaths, artifacts, transfers, scenario events). The anthropologist scripts read grid-world runs.
- `params.json`: the full configuration of the run.
- `artifacts.json`, `agent_names.json`, `costs.csv`, `food_counts.json`.
- `checkpoint_latest.pkl` and `checkpoint_previous.pkl`: what `--resume` restarts from.
- `frames/` and a video when `run.save_video` is on.
- `social_graph.jsonl` in social-graph runs: the full graph at the start and after every step. See [social_graph_replay.md](social_graph_replay.md).

## The AI Anthropologist

The anthropologist is an LLM-based analysis of a run. It has five steps: (1) it annotates each being's behaviour, (2) it builds the interaction graph and finds communities, (3) it annotates each group, (4) it scores artifacts for novelty and complexity and classifies them, (5) it reconstructs the artifact phylogeny. The steps and their outputs are described in [analysis_scripts/AI_ANTHROPOLOGIST.md](../analysis_scripts/AI_ANTHROPOLOGIST.md).

It runs in two ways.

**After a run.** The numbered scripts under `analysis_scripts/` run one step each over finished runs; `007_anthropologist.py` runs the whole pipeline. Set `EXPERIMENTS_NAMES` at the top of a script and run it from the repository root:

```bash
python analysis_scripts/007_anthropologist.py
```

The outputs go next to the run: `annotations/`, `community_annotations/`, `artifact_analysis/`.

**During a run.** `terralingua-anthropologist --exp_name <exp_name>` follows a run that has the remote API on, runs detection every few steps, and publishes field notes to the dashboard through Redis. It writes `logs/<exp_name>/field_notes.jsonl`. See [install_and_run.md](install_and_run.md).

The artifact complexity step needs the spaCy model: `python -m spacy download en_core_web_sm`.

## Notebooks

The notebooks under `analysis_scripts/notebooks/` show each step. `n000` gives the overall statistics of a run; `n001` to `n007` follow the pipeline, from per-being behaviour to an interactive phylogeny. They need the `analysis` extra:

```bash
pip install -e ".[analysis]"
jupyter notebook analysis_scripts/notebooks/
```

The notebooks read runs from `logs/` or `data/logs/` at the repository root. Set the experiment names in the first cells.

## The dataset

The runs of the paper are published at https://huggingface.co/datasets/GPaolo/TerraLingua, and a dashboard over them is at https://aianthropology.decisionai.ml/.
