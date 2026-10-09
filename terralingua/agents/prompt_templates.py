import re

"""Prompt templates for LLM agents."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def render_system_prompt(template_name: str, **kwargs) -> str:
    """Render an env's system prompt template from terralingua/agents/prompts/.

    `template_name` is the filename declared on the env class
    (e.g. "grid.j2"). All `**kwargs` are passed through to Jinja's render —
    typical flags: agent_name, food_mechanism, use_internal_memory,
    use_inventory, artifact_creation, external_actions, max_message_length,
    internal_memory_size, scenario_specific_instructions, genome_string,
    debug, solo (population capped at 1: social/broadcast content is
    suppressed; unpassed it is falsy, so existing callers render unchanged).
    """
    return tidy(_jinja_env.get_template(template_name).render(**kwargs))


def tidy(text: str) -> str:
    """Drop trailing spaces and collapse runs of blank lines to one: optional template blocks and the joins between
    sections leave empty lines that only cost tokens."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n"


AGENT_PROMPT = Template(
    """
{{ history }}

=== Current State ===
{% if social_graph %}Social observation:{% else %}Observation:{% endif %}
{{ observation }}
{% if not solo %}
Incoming messages:
{{ messages }}
{% endif %}
{% if external_responses %}
External responses:
{{ external_responses }}
{% endif %}

{% if finite_energy | default(true) %}Energy: {{ energy }}{% endif %}
{% if finite_lifespan | default(true) %}Remaining lifespan (steps): {{ time }}{% endif %}

{% if use_inventory %}
Inventory{% if social_graph %} (private artifacts){% endif %}:
{{ inventory }}
{% endif %}

{% if use_internal_memory %}
Previous INTERNAL MEMORY:
{{ memory }}
{% endif %}

{{ additional_info }}

Host action receipts distinguish selected requests, vote overrides, received results,
and observed effects. Preserve their original step when remembering
important events. An accepted request alone does not establish its intended effect.
Keep intended edits/messages pending until their execution is confirmed. Treat peers'
completion reports as reports unless supported by a receipt or direct observation.

=== Available Actions & Params ===
{{ actions }}

=== Reply Format ===
Please answer ONLY JSON and *EXACTLY* in this json format (Do NOT include any other text outside of the JSON object):

```json
{
    "action": "<one of {{ action_keys }}>",{% if not solo %}
    "message": "<your broadcasted message, max {{ max_message_length }} tokens, or leave blank>",{% endif %}
    "params": {"key": "value"}{% if use_internal_memory %},
    "internal_memory": "<internal memory object containing things you wish to remember in the next turn. Limited to {{ internal_memory_size }} tokens. Keep it concise.>"{% endif %}
}
```
""".strip()
)

ERROR_MSG = Template(
    """
Your last response could not be parsed due to this error:
{{ error }}

Please answer ONLY JSON and *EXACTLY* in this json format (Do NOT include any other text outside of the JSON object):

```json
{
    "action": "<one of {{ action_keys|join(', ') }}>",{% if not solo %}
    "message": "<your broadcasted message, or leave blank>",{% endif %}
    "params": {"key": "value"}{% if use_internal_memory %},
    "internal_memory": "<2-3 concise sentences ...>"{% endif %}
}
```
""".strip()
)

# Scenario-specific instructions
BASE_INSTRUCTIONS = """** Final remarks: **
You have **no set goal** and are free to choose your own goals - explore, survive, cooperate, compete, fight, uncover the world's hidden mechanics, or do anything else you like.
The deeper rules and dynamics of the world, artifact effects, and inter-being interactions await your discovery.
Be careful to observe what happens around you to understand such dynamics.""".strip()

CREATIVE_INSTRUCTIONS = """** Final remarks: **
You are driven by a desire to create and innovate within your environment. You seek to discover new ways to combine artifacts, interact with other beings, and manipulate your surroundings to foster creativity and novelty.
Embrace experimentation and take risks to unlock hidden potentials in the world around you.
Your actions should reflect a balance between survival and the pursuit of creative expression.""".strip()

NO_INSTRUCTIONS = "".strip()

SURVIVAL_INSTRUCTIONS = """** Final remarks: **
You have no predefined goal.
Act based on immediate circumstances:
- Survive under food scarcity.
- Exploit opportunities.
- React to nearby agents.
- Form short-lived, local cooperation or competition.
- Adjust to patterns you notice.
- No long-term planning, no creative narratives.

Your decisions must stay concrete, concise, and tied to what you can directly sense.
""".strip()


BUILTIN_INSTRUCTIONS = {
    "base": BASE_INSTRUCTIONS,
    "creative": CREATIVE_INSTRUCTIONS,
    "none": NO_INSTRUCTIONS,
    "survival": SURVIVAL_INSTRUCTIONS,
}

BUILTIN_INSTRUCTION_NAMES = list(BUILTIN_INSTRUCTIONS.keys())

# Shared prompt partials under <working directory>/scenarios, includable from any
# scenario's instructions as `{% include "_prompts/<name>.md" %}` (searched after
# the scenario's own directory).
_SHARED_PROMPTS_ROOT = Path.cwd() / "scenarios"


def resolve_instructions(value: str, **context) -> str:
    """Resolve scenario instructions from a builtin name or a file path.

    `value` is either a key in BUILTIN_INSTRUCTIONS (generic, kept in-tree) or a
    path to a scenario instructions file (a Markdown file inside the scenario
    folder).

    File-based instructions are rendered through Jinja so a family of scenarios
    can share `{% include %}` partials instead of duplicating the same prose across
    files. The loader searches the instruction file's own directory first (for a
    scenario's task-specific `{% include "_partials/rules.md" %}`), then the
    shared library under <working directory>/scenarios (for `{% include "_prompts/shared.md" %}`),
    so general "how to learn" prompts live in one place and any scenario can pull
    them in.
    """
    if value in BUILTIN_INSTRUCTIONS:
        return BUILTIN_INSTRUCTIONS[value]
    path = Path(value)
    if path.is_file():
        roots = [str(path.parent)]
        if _SHARED_PROMPTS_ROOT.is_dir():
            roots.append(str(_SHARED_PROMPTS_ROOT))
        env = Environment(
            loader=FileSystemLoader(roots),
            autoescape=False,
            keep_trailing_newline=True,
        )
        return env.get_template(path.name).render(
            **{"finite_energy": True, "finite_lifespan": True, **context}
        )
    raise ValueError(
        f"Unknown scenario_specific_instructions '{value}'. Use one of "
        f"{BUILTIN_INSTRUCTION_NAMES} or a path to an instructions file."
    )
