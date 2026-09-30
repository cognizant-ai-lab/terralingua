"""Checks for social action wording without changes to action rules."""

import copy

from terralingua.environment.actions import (
    ACTION_TEXT,
    build_deposit_artifact_action,
    build_retrieve_artifact_action,
    build_spawn_action,
    describe_social_actions,
)


def test_social_descriptions_preserve_shared_templates_costs_and_choices():
    templates_before = copy.deepcopy(ACTION_TEXT)
    actions = {
        name: copy.deepcopy(ACTION_TEXT[name])
        for name in ("give", "take", "create_artifact", "pickup_artifact", "drop_artifact", "give_artifact")
    }
    actions["create_artifact"]["description"] += " It costs 13 energy."
    actions["create_artifact"]["params"]["payload"] = "Text up to 745 tokens."
    for name, parameter in (("give", "target"), ("take", "target"), ("give_artifact", "target_agent")):
        actions[name]["params"][parameter] = {
            "description": actions[name]["params"][parameter],
            "choices": ["friend-b", "friend-a"],
            "metadata": {"keep": True},
        }
    original = copy.deepcopy(actions)

    result = describe_social_actions(actions)

    assert actions == original
    assert ACTION_TEXT == templates_before
    assert set(result) == set(actions)
    assert result["create_artifact"]["description"].endswith(" It costs 13 energy.")
    assert result["create_artifact"]["params"]["payload"] == "Text up to 745 tokens."
    assert result["create_artifact"]["params"]["type"] == actions["create_artifact"]["params"]["type"]
    assert result["create_artifact"]["params"]["movable"]["choices"] == ["true", "false"]
    for name in actions:
        assert set(result[name]) == set(actions[name])
        assert set(result[name]["params"]) == set(actions[name]["params"])
    for name, parameter in (("give", "target"), ("take", "target"), ("give_artifact", "target_agent")):
        target = result[name]["params"][parameter]
        assert target["choices"] == ["friend-b", "friend-a"]
        assert target["metadata"] == {"keep": True}
    for name in ("give", "take"):
        assert result[name]["params"]["amount"] == actions[name]["params"]["amount"]


def test_nested_shared_template_data_is_not_mutated():
    original = copy.deepcopy(ACTION_TEXT)
    actions = {"create_artifact": dict(ACTION_TEXT["create_artifact"])}
    assert actions["create_artifact"]["params"] is ACTION_TEXT["create_artifact"]["params"]

    result = describe_social_actions(actions)

    assert ACTION_TEXT == original
    assert result["create_artifact"]["params"] is not ACTION_TEXT["create_artifact"]["params"]


def test_optional_fixed_artifact_parameter_is_not_reintroduced():
    entry = copy.deepcopy(ACTION_TEXT["create_artifact"])
    entry["params"].pop("movable")

    result = describe_social_actions({"create_artifact": entry})

    assert "movable" not in result["create_artifact"]["params"]


def test_external_and_custom_action_wording_is_unchanged():
    external = {
        "description": "Move a node to a new map location.",
        "params": {"position": "Coordinates of the destination cell."},
        "input_schema": {"type": "object", "required": ["position"]},
        "optional": [],
    }
    custom = {"description": "Move energy between map locations.", "params": {"target": "Node ID"}}
    custom_spawn = {"description": "Create a node on a map.", "params": {"partner": {"description": "Map node"}}}
    actions = {"world_move": external, "give": custom, "spawn": custom_spawn}
    original = copy.deepcopy(actions)

    result = describe_social_actions(actions)

    assert result == original
    assert result["world_move"] is external
    assert result["give"] is custom
    assert result["spawn"] is custom_spawn


def test_spawn_partner_wording_preserves_eligibility_schema_and_cost():
    name, entry = build_spawn_action(19, 100, "sentence_directed", ["followed-peer"])
    original = copy.deepcopy(entry)

    result = describe_social_actions({name: entry})[name]

    assert entry == original
    assert result["description"] == original["description"]
    assert set(result["params"]) == set(original["params"])
    assert result["params"]["partner"]["choices"] == ["", "followed-peer"]
    for parameter in set(original["params"]) - {"partner"}:
        assert result["params"][parameter] == original["params"][parameter]
    assert "19 energy" in result["description"]
    assert "energy" in result["params"]
    assert "energy" in result["optional"]
    solo_name, solo_entry = build_spawn_action(19, 100, "sentence_directed", [])
    solo = describe_social_actions({solo_name: solo_entry})[solo_name]
    assert solo == solo_entry
    assert "partner" not in solo["params"]


def test_library_descriptions_avoid_spatial_wording():
    for builder in (build_deposit_artifact_action, build_retrieve_artifact_action):
        _, entry = builder(["guide", "notes"])
        for word in ("node", "location", "position"):
            assert word not in entry["description"]
            assert word not in entry["params"]["name"]["description"]
