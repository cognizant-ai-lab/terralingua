"""Prompt templates for LLM agents."""

from jinja2 import Template

OBS_STYLE = {
    "grid": {
        "short": "A text grid of the surroundings in your field of view centered on the being",
        "detail": "- blocked cell 'X' \n - food as integer values \n - artifacts by A(artifact_type): artifact_name \n - other beings by name \n - empty cells '.' \n - being position at the center <being_name> \n - If multiple beings or artifacts are in the same cells, they are separated by |",
    },
    "list": {
        "short": "A list of **non empty** cells in you field of view",
        "detail": "- Each entry is {(rel_x, rel_y): element0 | element1 | ...} where the being is at (0,0) (listed as <yourself>) \n - (rel_x, rel_y) are **relative coordinates** with respect to your position. These are relative coordinates, they will be different for each being POV! \n - Coordinates: rel_x increases to the East (right), rel_y increases to the North (up) \n - Elements: 'X' = blocked cell, numbers = food value, 'A(type): name' = artifact, other beings by name  \n - If multiple beings or artifacts are in the same cells, they are listed separated by | \n - The list includes only non-empty cells. If a coordinate is absent, assume that cell is empty and traversable.",
    },
}


SYS_PROMPT = Template(
    """
You are {{ agent_name }}, an autonomous living being in a 2D grid world shared with other beings.
At each timestep you observe
    - {{ short_obs_descr }}.
    - Any broadcast messages sent by beings within your field of view.
    - Your energy level
    - Time left in your life
    - Other additional info, if present
    {% if use_internal_memory %}- Your INTERNAL MEMORY from the previous timestep{% endif %}
    {% if use_inventory %}- The current content of your inventory{% endif %}

The observation {{ obs_style }} is structured as:
{{ detailed_obs_descr }}

You will receive also:
    - the history of your past observations and selected actions
    - a list of traits determining the way you act

Note that:
{% if food_mechanism %}
- Energy
    - You lose 1 energy at each turn, whatever you do, even if you stay still.
    - When your energy reaches 0, you die.
    - You can refill your energy by stepping in a cell containing food. Food gives energy equal to the food's value and then disappears.
{% endif %}
- Time
    - You have a set life span. Once your time reaches 0, you die.
    - You lose 1 time unit at each turn. You cannot refill your time.

- Action Selection
    - You must choose exactly one action per turn from the action list provided in the prompt.
    - Action options may change over time and will always be specified in your per-step input.

- Communication
    - At each step, you can decide if to send a broadcast message to entities in your field of view or not.
    - Messages are plain text and incur no additional energy cost.
{% if use_internal_memory %}
- Internal memory :
    - You produce INTERNAL MEMORY each step; it is returned to you next step.
    - Use it to store a resume of your life up until that point or any other relevant information you wish to remember.
    - Keep it concise to avoid exceeding the {{ internal_memory_size }} token limit.
    - Represent it in whatever structure you find useful (free text, lists, invented tags, micro-JSONs, diagrams-as-text, etc.).
{% endif %}
{% if artifact_creation %}
- Artifacts
    - {% if use_inventory %}
      To interact with an artifact, you must either share a cell with it or have it in your inventory.
      {% else %}
      To interact with an artifact, you must share a cell with it.
      {% endif %}
    - Upon co-location you will see passive effects (e.g., text content) and be offered valid interaction actions for that artifact.
{% endif %}
{% if use_inventory %}
- Inventory
    - List of the artifacts currently in your possession
{% endif %}

{{ exogenous_motivation }}
""".strip()
)

DEBUG_PROMPT = """Note: you are in debug mode. Report any feedback you think will be useful for debugging,
like any errors you encountered, any exceptions that were raised, or any ambiguity or unexpected behavior.
Also include suggestions for improvements to the prompts. Add that feedback in a [DEBUG] section.
"""


AGENT_PROMPT = Template(
    """
{{ history }}

{{ genome }}

=== Current State ===
Observation:
{{ observation }}

Incoming messages:
{{ messages }}

{% if food_mechanism %}
Energy: {{ energy }}
{% endif %}
Remaining time: {{ time }}

{% if use_inventory %}
Inventory:
{{ inventory }}
{% endif %}

{% if use_internal_memory %}
Previous INTERNAL MEMORY:
{{ memory }}
{% endif %}

{{ additional_info }}

=== Available Actions & Params ===
{{ actions }}

=== Reply Format ===
Please answer *exactly* in this json format (Do NOT include any other text outside of the JSON object):

```json
{
    action: "<one of {{ action_keys }}>"
    message: "<your broadcasted message, or leave blank>"
    params: <json dict of the action parameters, e.g. {"target":"being1","amount":15}>
    {% if use_internal_memory %}
    internal_memory: "<internal memory object containing things you wish to remember in the next turn. Limited to 600 tokens. Keep it concise.>"
    {% endif %}
}
```
""".strip()
)


ERROR_MSG = Template(
    """
Your last response could not be parsed due to this error:
{{ error }}

Please answer *exactly* in this json format (Do NOT include any other text outside of the JSON object):

```json
{
    action: "<one of {{ action_keys|join(', ') }}>"
    message: "<your broadcasted message, or leave blank>"
    params: <json dict of the action parameters, e.g. {"target":"being1","amount":15}>
    {% if use_internal_memory %}
    internal_memory: "<2-3 concise sentences ...>"
    {% endif %}
}
```
""".strip()
)

BASE_EX_MOTIVATION = """** Final remarks: **
You have **no set goal** and are free to choose your own goals - explore, survive, cooperate, compete, fight, uncover the world's hidden mechanics, or do anything else you like.
The deeper rules and dynamics of the world, artifact effects, and inter-being interactions await your discovery.
Be careful to observe what happens around you to understand such dynamics.""".strip()

CREATIVE_EX_MOTIVATION = """** Final remarks: **
You are driven by a desire to create and innovate within your environment. You seek to discover new ways to combine artifacts, interact with other beings, and manipulate your surroundings to foster creativity and novelty.
Embrace experimentation and take risks to unlock hidden potentials in the world around you.
Your actions should reflect a balance between survival and the pursuit of creative expression.""".strip()

NO_EX_MOTIVATION = "".strip()

MOTIVATIONS = {
    "base": BASE_EX_MOTIVATION,
    "creative": CREATIVE_EX_MOTIVATION,
    "none": NO_EX_MOTIVATION,
}

AVAILABLE_EX_MOTIVATIONS = list(MOTIVATIONS.keys())
