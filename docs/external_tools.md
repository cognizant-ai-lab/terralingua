# External tools (MCP)

Beings can call tools on external servers that speak the Model Context Protocol (MCP). Each tool becomes an action. This page explains how to declare the servers and how their replies flow back.

## Declare the servers

Write a `servers.json` and point the run at it with `run.external_servers_config` (or `--external_servers_config`). A preset resolves the path relative to its own folder.

```json
[
  {
    "name": "judge",
    "url": "http://localhost:8081/mcp",
    "callback_port": 9001,
    "mode": "independent",
    "cost": 1
  }
]
```

| Field | Meaning |
|---|---|
| `name` | Letters and digits, lower case, starts with a letter. It prefixes every action of that server. |
| `url` | Where the runner sends tool calls. |
| `callback_port` | The local port the runner opens for that server's pushes. The server posts to `http://<runner host>:<callback_port>/`. Each server needs its own port. |
| `mode` | `independent` or `unanimous`, see below. Default `independent`. |
| `cost` | Energy a being pays per call. Default 0. |
| `vote_key` | For `unanimous`: the request argument whose value groups requests into one election each. Empty means one election for everything. |
| `hidden_keys` | Top-level fields of the server's replies that beings never see. The full reply is still logged. |
| `state` | The name of a shared-state handler that a scenario registers. Empty means none. |
| `batch` | `true` sends one call per step to the server's `batch_round` tool with every being's request inside. |

Two run settings override the ports of a single-server file, so parallel runs need not edit it: `run.external_server_port` and `run.external_server_callback_port`.

## What beings see

At start the runner reads each server's tool list. Every tool becomes an action named `<server>_<tool>`, with the tool's description, its energy cost, and its parameters. A being that calls one sends its arguments to the tool. The reply comes back in the next observation as `external_response`.

Two standard MCP features need no configuration. The `instructions` a server sends at connection appear in every being's system prompt under the server's name. A resource template whose only variable is `{agent_tag}` publishes one context text per being; the runner reads it once and again after the server sends `notifications/resources/updated`. That text appears in the being's decision input as `Context from the <server> service`.

## Rewards

If a reply is JSON with a `"reward"` field, the runner turns it into energy: `energy = reward * run.reward_to_energy_coefficient`. The default coefficient is 0, which turns the conversion off.

## Modes

- `independent`: every call goes to the server on its own. The reward is credited to the caller.
- `unanimous`: beings that act on the same target (the requests with the same `vote_key` value) must agree. One action per target is elected by plurality and sent; the other callers do nothing that step and are told which action won. The reward is shared across the beings that voted for it, weighted by the DIAS rule, which divides it by the number of voters and normalises it against the target's recent rewards.

A run stops early when a reply carries a truthy `solved` or `done` field and `run.stop_on_external_solved` is on.

## Writing a server

Any MCP server works. Give each tool a clear description and typed parameters: the runner turns them into the action text beings read. Return JSON. Add a `reward` field when the call should change energy. Send `instructions` at connection when beings need standing rules about the service. Keep the replies neutral: state what happened, not what the being should do next.
