# Agent models

`agent.model` (or `--model`) sets the default model for LLM agents. The default is `claude-haiku-4-5`.

Use `agent.model_assignment` to give specific models to a chosen number of starting LLM agents. For example, this preset starts two Terra agents and three Luna agents.

```yaml
config:
  agent:
    model: gpt-6-luna
    model_assignment:
      gpt-6.1-terra: 2
  env:
    init_agents: 5
```

The equivalent CLI flags are `--init_agents 5 --model gpt-6-luna --model_assignment '{"gpt-6.1-terra":2}'`. Counts are nonnegative integers and their sum must fit the starting LLM population. Set `--run.seed` to reproduce which agents receive each model. Each agent keeps its model on resume and records it in its decision log.

By default, offspring inherit the model of the parent that took the `spawn` action, including in two-parent births. Set `agent.inherited_model: false` (CLI `--no-inherited_model`) to let that parent choose. The spawn action then requires an `offspring_model` from `agent.model` or any key in `agent.model_assignment`, including models with zero starting agents. With the example above, either parent can spawn either Terra or Luna. Automatic population replacements use the run default.

For agents driven by a dashboard user's API key, the child must use a model that the same provider can serve.

## Model keys

| Key | Provider | Notes |
|---|---|---|
| `claude-haiku-4-5` | Anthropic | Default. Fast and cheap. |
| `claude-sonnet-4-6` | Anthropic | |
| `claude-sonnet-5` | Anthropic | |
| `claude-sonnet-5-5` | Anthropic | |
| `claude-opus-4-8` | Anthropic | |
| `claude-opus-5-5` | Anthropic | |
| `o4-mini`, `o3-mini` | OpenAI | |
| `gpt-5.1`, `gpt-5-mini` | OpenAI | |
| `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6.1-terra` | OpenAI | |
| `QWEN2.5` | Local (vLLM) | Qwen2.5-32B-Instruct |
| `QWEN3` | Local (vLLM) | Qwen3-32B |
| `DeepSeek-R1-32` | Local (vLLM) | DeepSeek-R1-Distill-Qwen-32B. The paper's model. |
| `DeepSeek-R1-70` | Local (vLLM) | DeepSeek-R1-Distill-Llama-70B |

Anthropic models need `ANTHROPIC_API_KEY` in `.env`. OpenAI models need `OPENAI_API_KEY`. The list of keys is `MODEL_MAP` in `terralingua/experiment/llm_router.py`; a name that starts with `claude-` goes to Anthropic, a name that starts with `gpt-` or looks like `o3`, `o4` goes to OpenAI; a name of the form `provider/model` goes to litellm as given.

Two settings shape a call: `agent.max_tokens` caps the reply length (each provider has its own default), and `agent.reasoning_effort` sets the reasoning level where the model supports it (`minimal` to `max`). The default is `low`. Set it to `null` to use the provider's own default.

## Agent prompt caching

Agent caching is automatic for direct Anthropic and OpenAI routes, including
native model names and explicit `anthropic/model` or `openai/model` names.
Stable system instructions, server instructions and MCP tool definitions form
the cached prefix. Current observations, tool availability and rolling history
remain in the user message. History limits and the agent's decision loop do not
change.

Anthropic uses an explicit `cache_control` boundary with its default renewable
five-minute lifetime. Supported OpenAI models use an explicit boundary and a
thirty-minute lifetime; older OpenAI models retain automatic provider caching.
String and text-block instructions both work. Explicit caller-supplied cache
options are preserved. Custom OpenAI-compatible endpoints are not assumed to
implement OpenAI's cache API.

Each new model/prefix needs warming. A hit requires an identical prefix, enough
tokens for that model and an available cache entry. In particular, Haiku 4.5
requires 4,096 tokens, while Sonnet 5.5 and Opus 5.5 require 512. A short prefix
can therefore work on one model and miss on another. TL does not pad instructions
to satisfy a threshold. See the [Anthropic caching guide](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
and [OpenAI caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
for current model limits and prices.

`costs.csv` and agent action records include `cached_input_tokens` and
`cache_write_input_tokens`. These are subsets of `input_tokens`, not additional
tokens. Missing telemetry stays unknown rather than being counted as zero;
historical CSV rows retain blank cache counts. Costs use LiteLLM's provider
pricing, including cache writes and reads.

To check caching in an experiment, inspect these counters after successive
decisions using the same model and instruction prefix. A positive
`cached_input_tokens` value confirms a cache read; `cache_write_input_tokens`
records writes when the provider reports them. Compare cached input tokens with
total input tokens to see how much of the prompt was reused. An empty counter
means the provider did not supply that telemetry. A zero read count can reflect
warming, a short or changed prefix, or an expired entry.

## Local models with vLLM

Local models need a running [vLLM](https://github.com/vllm-project/vllm) server. Start one or more on the ports the run expects. The default ports are `9000` to `9003` and `9010` to `9012`.

```bash
vllm serve Qwen/Qwen3-32B --port 9000
```

Then run with the model key and the ports:

```bash
terralingua grid_baseline --model QWEN3 --ports "[9000, 9001]"
```

At start the runner asks each port which model it serves, keeps the ports that serve the requested model, and spreads the calls over them.

## The anthropologist's model

The live anthropologist takes its own `--model` (default `claude-haiku-4-5`). The offline analysis scripts set their model at the top of each file.
