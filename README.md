# TerraLingua

**Paper:** [ResearchGate](https://www.researchgate.net/publication/402263491_TerraLingua_Emergence_and_Analysis_of_Open-endedness_in_LLM_Ecologies) · [arXiv](https://arxiv.org/abs/2603.16910)

**Dataset:** https://huggingface.co/datasets/GPaolo/TerraLingua

**Dataset dashboard:** https://aianthropology.decisionai.ml/

![TerraLingua agents](assets/environment.gif)

A multi-agent simulation framework for studying emergent behavior, artifact creation, and cultural evolution.

LLM-powered agents interact in a shared grid or graph world: they forage for resources, create text artifacts, reproduce, and communicate. This enables research into how language-using agents develop social structure and culture over time.

The **AI Anthropologist**, itself an LLM agent, analyzes the simulation logs, during a run or after it, to annotate agent behaviors, infer group dynamics, classify artifacts, and trace cultural lineages. It gives a qualitative and quantitative account of what emerged.

The figure below shows the TerraLingua system and the AI Anthropologist.

![TerraLingua and the AI Anthropologist](assets/whole.png)

## Installation

Requires **Python 3.10+**. Runs that save videos also need **ffmpeg** (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian and Ubuntu).

```bash
python -m venv .venv && source .venv/bin/activate     # or: conda create -n terralingua python=3.12 && conda activate terralingua
pip install -e .                          # the package and the commands terralingua, terralingua-dashboard, terralingua-anthropologist
pip install -e ".[analysis]"              # optional: the libraries the notebooks use
python -m spacy download en_core_web_sm   # once: the anthropologist's artifact analysis needs it
```

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

The dashboard, the live anthropologist, and the Docker demo are described in [docs/install_and_run.md](docs/install_and_run.md).

## Running the paper experiments

Each experimental condition of the paper is a preset under [scenarios/paper/](scenarios/paper/). Run one from the repository root:

```bash
terralingua paper_core
```

| Preset | Condition |
|---|---|
| `paper_core` | Baseline |
| `paper_abundant` | Long history and food everywhere |
| `paper_artifact_cost` | Creating an artifact costs energy |
| `paper_creative` | Creative motivation text |
| `paper_inert_artifacts` | Artifacts can be created but not read or used |
| `paper_long_memory` | Long history |
| `paper_no_motivation` | No motivation text |
| `paper_no_personality` | No personality traits |

`paper_core` is the reference configuration of the paper. Logs are written to `logs/<exp_name>/`. The paper used `DeepSeek-R1-32` served locally with vLLM; pass `--model claude-haiku-4-5`, or another key from [docs/models.md](docs/models.md), to run with an API model. Any setting can be overridden on the command line, for example `terralingua paper_core --max_ts 500`. `terralingua --list` shows every preset and `terralingua --help` every setting; [template_preset.yaml](template_preset.yaml) lists every setting with its default and a comment, so copy the keys you change into a `<name>.preset.yaml`. See [docs/configuration.md](docs/configuration.md). [scenarios/paper/README.md](scenarios/paper/README.md) lists how the presets map to the original scripts.

## Data analysis and visualization

The **AI Anthropologist** annotates agent behaviors, infers group dynamics, classifies artifacts, and traces cultural lineages. [analysis_scripts/AI_ANTHROPOLOGIST.md](analysis_scripts/AI_ANTHROPOLOGIST.md) describes the pipeline in detail. [docs/analysis.md](docs/analysis.md) describes what a run writes and how to run the anthropologist during a run.

The scripts follow a numbered order and run from the repository root. Set `EXPERIMENTS_NAMES` at the top of a script first.

| Script | Description |
|---|---|
| `001_llm_agent_analyser.py` | Annotate agent logs with LLM-generated behavior labels |
| `002_make_graph.py` | Build interaction graphs and compute network metrics |
| `003_llm_group_analyser.py` | Group-level behavioral analysis |
| `004_artifact_analysis.py` | Compute artifact complexity metrics |
| `005_artifact_classification.py` | Classify artifacts into behavioral categories |
| `006_artifact_philogeny.py` | Analyze artifact genealogy and conceptual ancestry |
| `007_anthropologist.py` | Run the whole pipeline over a finished run |

```bash
python analysis_scripts/001_llm_agent_analyser.py
```

Notebooks in `analysis_scripts/notebooks/` mirror the pipeline. They need the `analysis` extra.

| Notebook | Description |
|---|---|
| `n000_general_stats.ipynb` | Overall experiment statistics |
| `n001_llm_agent_analyser.ipynb` | Per-agent behavior visualization |
| `n002_graph_analysis.ipynb` | Interaction network plots |
| `n003_llm_group_analysis.ipynb` | Group dynamics |
| `n004_artifact_analysis.ipynb` | Artifact complexity over time |
| `n005_artifact_categories.ipynb` | Classification results |
| `n006_artifact_phylogeny.ipynb` | Artifact lineage trees |
| `n007_interactive_phylogeny.ipynb` | Interactive phylogeny explorer |

```bash
jupyter notebook analysis_scripts/notebooks/
```

## Creating your own scenario

A scenario allows you to run a specific setup in TerraLingua. It is one Python package with a few hooks, plus a preset. [docs/writing_a_scenario.md](docs/writing_a_scenario.md) is the guide, and [scenarios/example/](scenarios/example/) is a small complete scenario to copy:

```bash
terralingua example
```

Further guides: [external tools over MCP](docs/external_tools.md), [agent models](docs/models.md), [configuration](docs/configuration.md), [seeding a graph world from a Neuro-SAN network](docs/neuro_san_hocon.md), [social graph recordings and replay](docs/social_graph_replay.md).

## Citation

If you use TerraLingua in your research, please cite:

```bibtex
@techreport{paolo26terralingua,
title = "TerraLingua: Emergence and Analysis of Open-Endedness in LLM Ecologies",
author = "Giuseppe Paolo and Jamieson Warner and Hormoz Shahrzad and Babak Hodjat and Risto Miikkulainen and Elliot Meyerson",
year = 2026,
month = jan,
institution = "Cognizant AI Lab",
url = "https://arxiv.org/abs/2603.16910",
doi = "10.48550/arXiv.2603.16910",
number = "2026-01",
}
```
