# Social graph recordings and replay

## Recording

Every `social_graph` run writes `social_graph.jsonl` in its experiment log
directory automatically. No extra flag or dashboard connection is required.
This includes every social-graph scenario.

A snapshot captures the initial state and then the complete graph after each
finished timestep, including population changes. Each contains all current
nodes, directed follow edges, pending connection requests, and agent identifiers.
Two agents following each other produce two directed follow edges. Pending
requests are recorded separately from established connections.

These are timestep snapshots: a connection created and removed within one
timestep is not visible in the completed state. The recording does not contain
message bodies or artifact contents.

## Dashboard

Use the replay controls in the existing dashboard graph pane to load the
connected run's recording or open a saved `social_graph.jsonl` file from your
computer. Loading a recording freezes its replay range; load it again to include
newly completed steps.

Play/pause, previous/next step, the timeline slider, and playback speed let you
inspect how the graph changes. Click a node to inspect its recorded identity and
connections. Node positions remain fixed across a loaded recording so births,
deaths, and connection changes do not rearrange the other nodes.

Replay changes only the graph pane. The rest of the dashboard still shows the
live run. Return to Live to see the current graph again.

Video export records the graph canvas with the recorded step number. The browser
selects a supported video format; frames are rendered at 1000 × 1000 pixels. Export uses the selected playback speed and
runs in real time; keep the tab visible until it finishes.

Current-run loading uses the existing dashboard session and reads the runner's
configured log directory. The API server needs access to that directory. A
headless run can instead be opened as a saved file. Old runs without a graph
recording cannot be replayed reliably from their agent observations.

## Format and resuming

The file is newline-delimited JSON, one snapshot per row:

- `schema_version`: recording format version (currently 1).
- `run_id`: experiment run identifier, preserved by checkpoint resumes.
- `segment_id`: identifier for one runner invocation.
- `segment_start`: whether this snapshot begins that invocation.
- `step`: environment timestep.
- `graph`: nodes, edges, agents, and layout data used by the dashboard.

Edges use `source`, `target`, and `type` (`follow` or `pending`). Agent
records associate stable tags and names with graph node identifiers.

The file preserves append order, including superseded history after a checkpoint
rollback. For a single coherent timeline, use
`terralingua.experiment.social_graph_history.read_social_graph_history(path)`.
The dashboard applies the same rules:

1. A different run ID starts a new timeline.
2. A segment-start snapshot replaces earlier snapshots at its step and later
   steps, retaining the earlier prefix.
3. An incomplete final line from an interrupted write is ignored. Corruption
   inside the file is reported.

The returned `snapshots` list is suitable for further graph analysis or custom
plots. Recording errors are surfaced rather than silently losing history.

## Development checks

Run the Python recording and API tests with the project's Python environment:

```bash
python -m pytest -q tests/test_social_graph_history.py tests/test_graph_history_api.py tests/test_runner_completion.py
```

The dashboard's dependency-free JavaScript tests use a recent Node runtime
(Node 22 or later):

```bash
node --test tests/test_graph_replay.mjs
```
