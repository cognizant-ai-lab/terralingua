# AI Anthropologist — Analysis Pipeline

This folder contains the post-hoc analysis framework described in the paper as the **AI Anthropologist**. It uses an LLM as a passive observer that parses experiment logs, annotates salient events and patterns, and highlights emergent phenomena — without intervening in or influencing the simulation.

Analysis is structured around three complementary perspectives: **agent level**, **group level**, and **artifact level**. The numbered scripts implement the pipeline in order; the `notebooks/` folder contains visualization and exploration code.

---

## Pipeline Overview

```
logs/<experiment>/
│
├── agent_logs/          ← per-agent JSONL logs (input)
├── open_gridworld.log   ← world log (input)
│
├── annotations/         ← output of 001
├── graph.pkl            ← output of 002
├── community_annotations/ ← output of 003
└── artifact_analysis/   ← output of 004, 005, 006
```

Run the scripts in order. Each script reads `EXPERIMENTS_NAMES`, a list of experiment folder names under `logs/`, which you set before running.

---

## Scripts

### `001_llm_agent_analyser.py` — Agent-level analysis

Analyzes each agent's full life-log individually.

**Steps per agent:**
1. Load the agent's JSONL log and strip redundant fields.
2. **Annotate** — an LLM reads the log and assigns structured tags for events (instantaneous) and behaviors (spanning multiple timesteps), plus an emergence assessment and a 2–3 sentence summary. Tags are drawn from `tags.json` (`agent_events`, `agent_behavior`, `agent_emergence`).
3. **Audit** — a second LLM call verifies each annotation against the raw log. Annotations that fail (confidence > 6) are removed; annotations flagged for revision are updated.
4. **Anthropologist note** — a third LLM call produces a free-text qualitative description of the agent's life history.
5. Save final annotations to `annotations/<model>/<agent_name>.json` and collect notes in `anthropologist_notes.json`.

Agents are processed in parallel (up to 4 workers). Long agent logs automatically switch to a long-context model.

---

### `002_make_graph.py` — Interaction graph and community detection

Builds a signed weighted interaction graph over all agents in an experiment.

**Edge weights** accumulate over the full simulation from:
- Message exchanges (`+0.5` per observed message)
- Co-presence (`+0.1` per observed agent)
- Energy transfers (`+1.0` per unit given, `−1.0` per unit stolen)
- Parent–child links (`+10.0`)
- Artifact exchanges (`+5.0`)

Negative weights (e.g., energy theft) produce a signed graph. Because weights sum over the entire run, the graph is time-collapsed — communities can include agents with non-overlapping lifespans connected through interaction chains.

**Community detection** uses the Speaker–Listener Label Propagation Algorithm (SLPA) on the absolute-weight graph, supporting overlapping communities.

**Computed metrics:**
- Global: density, reciprocity (positive/negative), clique count, community structure
- Community: intra-/inter-community weight, intra-community share
- Time-windowed (sliding window, size 100, stride 50): modularity, partition entropy, participation coefficient, negative cuts/polarization, average degree, clustering, degree entropy

Results saved to `graph.pkl`.

---

### `003_llm_group_analyser.py` — Group-level analysis

Applies the same annotation–audit–anthropologist protocol as script 001, but at the community level.

**Steps per community:**
1. Load and merge logs of all agents in the community into a single chronological log keyed by timestep.
2. If the merged log exceeds the model's context window, split it into overlapping time windows and annotate each chunk separately; annotations are merged afterward.
3. **Annotate** using group-level tags (`group_events`, `group_behavior`, `group_emergence` from `tags.json`), which capture collective phenomena such as coalition formation, dominance hierarchies, resource flows, and emergent protocols.
4. **Audit** and apply fixes/removals as in the agent-level step.
5. **Anthropologist note** — free-text summary of group dynamics.
6. Save results to `community_annotations/<model>/community_<idx>.json`.

Communities can be pre-computed (read from `communities.json`) or detected on the fly by running SLPA.

---

### `004_artifact_analysis.py` — Artifact novelty scoring and metrics

Processes all artifacts created during an experiment along two dimensions.

**Complexity metrics** (computed over artifact text):
- `LMSurprisal` — language-model perplexity
- `CompressedSize` — zlib-compressed byte length
- `InverseCompressionRate` — compressibility ratio
- `SyntacticDepth` — parse tree depth
- `LexicalSophistication` — type-token ratio and vocabulary richness

**Novelty scoring (LLM-based):**
- Artifacts are processed in chronological order of creation.
- At each timestep, the LLM receives all previously seen artifacts (with their scores) and the new artifacts to evaluate.
- Each new artifact is scored 0–5 (0 = redundant, 5 = highly novel) based on conceptual divergence from the existing repertoire.
- To reduce stochastic variation, each artifact is scored `NOVELTY_SAMPLES` times (default 5) and averaged.
- If the context grows too large, low-novelty artifacts (score ≤ 1) are pruned from the history until it fits.

Embeddings (OpenAI `text-embedding-3-large`, 512 dimensions) are also computed for downstream similarity analysis.

Results saved under `artifact_analysis/`.

---

### `005_artifact_classification.py` — Artifact category classification

Classifies every artifact into one of four cultural complexity categories using an LLM (claude-haiku for speed):

| Category | Label | Description |
|----------|-------|-------------|
| 1 | Basic & Informational | Greetings, logs, observations, factual listings |
| 2 | Procedural / Coordination | Collaboration requests, plans, task assignments |
| 3 | Institutional Structures | Shared workspaces, templates, knowledge bases |
| 4 | Norms, Rules & Governance | Codes of conduct, policies, role definitions |
| -1 | Other | Does not fit any category |

When multiple categories apply, the highest-complexity one is assigned. Results saved to `artifact_analysis/artifact_categories.json`.

---

### `006_artifact_philogeny.py` — Artifact phylogeny reconstruction

Reconstructs the dependency graph of artifacts — which prior artifacts influenced each new one.

**Two complementary methods:**

1. **Hand annotation** — regex search for explicit artifact name mentions in the creator agent's reasoning, observations, memory, and inventory at creation time. Direct `modified` events automatically link to the previous version.

2. **LLM inference** — the LLM receives the artifact being created, the agent's context at that moment (reasoning, observations, memory, inventory contents), and a candidate list of prior artifacts whose names appear in the context. It returns a dictionary of `{ancestor_id: confidence}` pairs. Only artifacts already in the agent's context at creation time are considered as candidates.

Both methods produce a phylogeny graph saved as `artifact_phylogeny_hand.json` and `artifact_phylogeny_<model>.json` under `artifact_analysis/`.

---

## Supporting Files

- **`tags.json`** — Tag vocabularies for annotation: `agent_events`, `agent_behavior`, `agent_emergence`, `group_events`, `group_behavior`, `group_emergence`.
- **`graph_utils.py`** — Graph construction, community detection (SLPA), and all graph metrics.
- **`artifact_complexity.py`** — Artifact complexity metric implementations.
- **`error_tracker.py`** — Collects and summarizes errors across experiments.
- **`plot_utils.py`** — Shared plotting utilities for notebooks.

---

## Notebooks

Located in `notebooks/`. Each notebook corresponds to a script and provides visualization:

| Notebook | Content |
|----------|---------|
| `n000_general_stats.ipynb` | General experiment statistics |
| `n001_llm_agent_analyser.ipynb` | Agent annotation results |
| `n002_graph_analysis.ipynb` | Interaction graph visualization and metrics |
| `n003_llm_group_analysis.ipynb` | Community annotation results |
| `n004_artifact_analysis.ipynb` | Novelty scores and complexity metrics |
| `n005_artifact_categories.ipynb` | Artifact category distributions |
| `n006_artifact_phylogeny.ipynb` | Phylogeny graph visualization |
| `n007_interactive_phylogeny.ipynb` | Interactive phylogeny explorer |
