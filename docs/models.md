# Agent models

`agent.model` (or `--model`) names the model every being uses to decide. The default is `claude-haiku-4-5`.

## Model keys

| Key | Provider | Notes |
|---|---|---|
| `claude-haiku-4-5` | Anthropic | Default. Fast and cheap. |
| `claude-sonnet-4-6` | Anthropic | |
| `claude-sonnet-5` | Anthropic | |
| `claude-opus-4-8` | Anthropic | |
| `claude-opus-5-5` | Anthropic | |
| `o4-mini`, `o3-mini` | OpenAI | |
| `gpt-5.1`, `gpt-5-mini` | OpenAI | |
| `QWEN2.5` | Local (vLLM) | Qwen2.5-32B-Instruct |
| `QWEN3` | Local (vLLM) | Qwen3-32B |
| `DeepSeek-R1-32` | Local (vLLM) | DeepSeek-R1-Distill-Qwen-32B. The paper's model. |
| `DeepSeek-R1-70` | Local (vLLM) | DeepSeek-R1-Distill-Llama-70B |

Anthropic models need `ANTHROPIC_API_KEY` in `.env`. OpenAI models need `OPENAI_API_KEY`. The list of keys is `MODEL_MAP` in `terralingua/experiment/llm_router.py`; a name that starts with `claude-` goes to Anthropic, a name that starts with `gpt-` or looks like `o3`, `o4` goes to OpenAI; a name of the form `provider/model` goes to litellm as given.

Two settings shape a call: `agent.max_tokens` caps the reply length (each provider has its own default), and `agent.reasoning_effort` sets the reasoning level where the model supports it (`minimal` to `max`).

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
