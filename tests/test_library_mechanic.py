"""The shared library mechanic: menus, deposit and retrieve, the metadata listing,
keyword search, in-place updates with versions and the first-wins rule, delete,
the size cap, change notifications, the newborn listing, what dies with a being,
aging, checkpoints, inert artifacts, and a grid world host."""

import numpy as np
import pytest

from terralingua.config.models import GraphConfig
from terralingua.environment.actions import LocationAffordance
from terralingua.environment.grid_env import OpenGridWorld
from terralingua.environment.social_graph_env import OpenSocialGraphWorld
from terralingua.mechanics import library as texts
from terralingua.mechanics.library import Library, Options

WORLD = dict(
    food_mechanism=False, energy_death=False, init_food=0, food_spawn_rate=0,
    food_decay_rate=0, init_agent_energy=100, lifespan=200, headless=True,
    use_inventory=True, artifact_creation_cost=0,
)
BEINGS = [("a", "Alice", "0"), ("b", "Bob", "1"), ("c", "Carol", "2")]


def social_world(tmp_path, topology="complete", **options):
    graph = GraphConfig.complete(n_nodes=4) if topology == "complete" else GraphConfig.ring(n_nodes=4)
    env = OpenSocialGraphWorld(graph_cfg=graph, log_path=tmp_path, **WORLD)
    env.attach(Library(Options(**options)))
    for tag, name, pos in BEINGS:
        env.add_agent(tag, name, "no_traits", position=pos)
    env.restart_env(seed=3, agent_poses={tag: pos for tag, _, pos in BEINGS})
    return env


def library_of(env) -> Library:
    return env.mechanics[0]


def seed(env, pose, name, payload="some knowledge", movable=True, lifespan=-1, creator="a"):
    env.add_artifact(pose=pose, art_type="text", art_name=name, payload=payload,
                     creator=creator, lifespan=lifespan, movable=movable)


def act(env, actions: dict):
    for tag in actions:
        env._get_avail_actions(tag)
    moves = {tag: {"action": action, "messages": {}, "params": params} for tag, (action, params) in actions.items()}
    return env.step(moves)


def outcome(env, tag, action, **params) -> str:
    infos = act(env, {tag: (action, params)})[4]
    return infos[tag]["Action outcome"]


def deposited(tmp_path, **options):
    env = social_world(tmp_path, **options)
    seed(env, "0", "recipe", payload="Bake the bread at high heat for twenty minutes.")
    assert outcome(env, "a", "deposit_artifact", name="recipe") == texts.DEPOSITED.format(name="recipe")
    return env


def test_menu_follows_the_library_state_and_exclusions(tmp_path):
    env = social_world(tmp_path)
    assert set(env._get_avail_actions("a")) & set(texts.ACTIONS) == {"show_library_content"}
    seed(env, "0", "recipe")
    assert "deposit_artifact" in env._get_avail_actions("a")
    assert "deposit_artifact" not in env._get_avail_actions("b")
    outcome(env, "a", "deposit_artifact", name="recipe")
    menu = env._get_avail_actions("b")
    assert set(texts.ACTIONS) - {"deposit_artifact"} <= set(menu)
    assert menu["read_artifact"]["params"]["name"] == texts.ACTIONS["read_artifact"]["params"]["name"]
    assert menu["create_artifact"]["description"].endswith(texts.CREATE_ARTIFACT_NOTE)
    env.excluded_actions = {"delete_artifact", "show"}
    env.add_agent_affordance("b", LocationAffordance(action="read_artifact", mode="remove"))
    menu = env._get_avail_actions("b")
    assert "delete_artifact" not in menu and "read_artifact" not in menu
    assert "show_library_content" in menu and "show_versions" in menu
    assert "read_artifact" in env._get_avail_actions("c")


def test_deposit_detaches_the_artifact_and_the_listing_shows_metadata(tmp_path):
    env = deposited(tmp_path)
    assert "recipe" not in env.pos_artifacts["0"]
    assert env.artifact_location["recipe"] == ("stored", "library")
    assert library_of(env).stored(env) == ["recipe"]
    listing = outcome(env, "b", "show_library_content")
    assert listing.startswith(texts.LISTING_HEADER.format(count=1))
    line = listing.splitlines()[1]
    for part in ("recipe", "by Alice", "creation step 0", texts.ACCESSED_NONE, "tokens", texts.EXPIRY_NONE):
        assert part in line
    assert outcome(env, "b", "read_artifact", name="recipe").startswith("recipe (v0):\nBake the bread")
    assert "accessed by 1 other being(s), last step 2" in outcome(env, "c", "show_library_content")
    assert outcome(env, "c", "show_library_content", creator="Bob") == texts.LISTING_NO_MATCH
    assert "recipe" in outcome(env, "c", "show_library_content", creator="Alice", since=0)


def test_search_ranks_by_keywords_and_shows_a_passage(tmp_path):
    env = deposited(tmp_path)
    seed(env, "1", "ledger", payload="Cash 100, loan 300, income this year 42.", creator="b")
    outcome(env, "b", "deposit_artifact", name="ledger")
    result = outcome(env, "c", "search_library", query="bread heat")
    lines = result.splitlines()
    assert lines[0] == texts.SEARCH_HEADER.format(query="bread heat", count=1)
    assert lines[1].startswith("recipe |")
    assert "Bake the bread" in lines[2]
    assert outcome(env, "c", "search_library", query="volcano") == texts.SEARCH_NONE.format(query="volcano")
    assert outcome(env, "c", "search_library", query="cash", creator="Alice") == texts.SEARCH_NONE.format(query="cash")


def test_update_keeps_versions_and_the_first_update_of_a_step_wins(tmp_path):
    env = deposited(tmp_path)
    assert outcome(env, "b", "update_artifact", name="recipe", payload="Bake at low heat.") == texts.UPDATED.format(name="recipe", version=1)
    assert env.artifacts["recipe"].payload == "Bake at low heat."
    versions = outcome(env, "c", "show_versions", name="recipe")
    assert versions.startswith(texts.VERSIONS_HEADER.format(name="recipe", count=1))
    assert "by Alice | replaced by Bob" in versions and "high heat" in versions
    assert "last change step 1 by Bob" in outcome(env, "c", "show_library_content")
    infos = act(env, {"b": ("update_artifact", {"name": "recipe", "payload": "Bob's text"}),
                      "c": ("update_artifact", {"name": "recipe", "payload": "Carol's text"})})[4]
    results = {tag: infos[tag]["Action outcome"] for tag in ("b", "c")}
    winners = [tag for tag, text in results.items() if text.startswith("Artifact recipe updated")]
    losers = [tag for tag, text in results.items() if text.startswith("Refused.")]
    assert len(winners) == 1 and len(losers) == 1
    assert env.artifacts["recipe"].version == 2
    assert env.artifacts["recipe"].payload in results[losers[0]]


def test_retrieve_and_delete(tmp_path):
    env = deposited(tmp_path)
    assert outcome(env, "b", "retrieve_artifact", name="recipe") == texts.RETRIEVED.format(name="recipe")
    assert "recipe" in env.pos_artifacts["1"] and env.artifact_location["recipe"] == ("map", "1")
    assert library_of(env).stored(env) == []
    outcome(env, "b", "deposit_artifact", name="recipe")
    assert outcome(env, "c", "delete_artifact", name="recipe") == texts.DELETED.format(name="recipe")
    assert "recipe" not in env.artifacts
    assert any(art.name == "recipe" and art.deletion_time == env.step_count - 1 for art in env.expired_artifacts)
    assert outcome(env, "b", "show_library_content") == texts.LISTING_EMPTY


def test_a_name_that_is_not_text_is_refused(tmp_path):
    env = deposited(tmp_path)
    refused = texts.NOT_IN_LIBRARY.format(name=["recipe"])
    assert outcome(env, "b", "update_artifact", name=["recipe"], payload="x") == refused
    for action in ("delete_artifact", "retrieve_artifact", "read_artifact"):
        assert outcome(env, "b", action, name=["recipe"]) == refused


def test_a_full_library_refuses_deposits_and_names_candidates(tmp_path):
    env = deposited(tmp_path, max_size=1)
    seed(env, "1", "second")
    refusal = outcome(env, "b", "deposit_artifact", name="second")
    assert refusal.startswith("Refused.") and "recipe" in refusal
    assert "second" in env.pos_artifacts["1"]


@pytest.mark.parametrize("mode, readers", [("all", {"b", "c"}), ("followers", {"b"}), ("none", set())])
def test_change_notifications_reach_the_configured_beings(tmp_path, mode, readers):
    env = social_world(tmp_path, topology="ring", notifications=mode)
    seed(env, "0", "recipe")
    infos = act(env, {"a": ("deposit_artifact", {"name": "recipe"})})[4]
    told = {tag for tag in ("a", "b", "c") if texts.NOTE_CHANGES in infos[tag]}
    assert told == readers
    for tag in readers:
        assert infos[tag][texts.NOTE_CHANGES].startswith("Alice deposited recipe: recipe |")


def test_a_newborn_gets_the_listing_in_its_first_observation(tmp_path):
    env = deposited(tmp_path)
    _, info = env.add_agent("d", "Dan", "no_traits", position="3")
    assert info[texts.NOTE_LIBRARY].startswith(texts.LISTING_HEADER.format(count=1))
    assert "recipe" in info[texts.NOTE_LIBRARY]
    env = deposited(tmp_path / "child")
    infos = act(env, {"a": ("spawn", {"name": "Kid"})})[4]
    child = infos["a"]["spawn"]["child_tag"]
    assert "recipe" in infos[child][texts.NOTE_LIBRARY]


def test_what_dies_with_a_being_enters_the_library(tmp_path):
    env = social_world(tmp_path)
    seed(env, "1", "held")
    env.stow_artifact("held", "b")
    seed(env, "1", "onnode")
    env._kill("b")
    assert set(library_of(env).stored(env)) == {"held", "onnode"}
    assert env.artifact_location["held"] == ("stored", "library")
    assert "1" not in env.world_graph.all_nodes()
    env_full = social_world(tmp_path / "full", max_size=0)
    seed(env_full, "1", "doomed")
    env_full._kill("b")
    assert "doomed" not in env_full.artifacts
    assert any(art.name == "doomed" for art in env_full.expired_artifacts)
    assert env_full.get_state_ckpt()["pending_deaths"][0]["artifacts"] == ["doomed"]


def test_stored_artifacts_age_and_expire(tmp_path):
    env = social_world(tmp_path)
    seed(env, "0", "ephemeral", lifespan=2)
    outcome(env, "a", "deposit_artifact", name="ephemeral")
    for _ in range(3):
        act(env, {"a": ("noop", {})})
    assert "ephemeral" not in env.artifacts
    assert any(art.name == "ephemeral" for art in env.expired_artifacts)


def test_the_library_survives_a_checkpoint(tmp_path):
    env = deposited(tmp_path)
    outcome(env, "b", "read_artifact", name="recipe")
    outcome(env, "b", "update_artifact", name="recipe", payload="Bake at low heat.")
    state = env.get_state_ckpt()
    env2 = social_world(tmp_path / "again")
    env2.set_state_ckpt(state)
    assert library_of(env2).stored(env2) == ["recipe"]
    assert env2.artifact_location["recipe"] == ("stored", "library")
    listing = outcome(env2, "c", "show_library_content")
    assert "accessed by 1 other being(s)" in listing
    assert "last change step 2 by Bob" in listing


def test_inert_artifacts_leave_only_the_listing(tmp_path):
    env = deposited(tmp_path)
    env._get_avail_actions("b")
    env.inert_artifacts = True
    info = env.step({"b": {"action": "read_artifact", "messages": {}, "params": {"name": "recipe"}}})[4]["b"]
    assert info["Action outcome"] == texts.INERT
    assert set(env._get_avail_actions("b")) & set(texts.ACTIONS) == {"show_library_content"}
    assert "recipe" in outcome(env, "c", "show_library_content")


def test_the_order_of_a_steps_actions_is_random(tmp_path):
    orders = set()
    for seed_value in range(6):
        env = social_world(tmp_path / str(seed_value))
        env.rng = np.random.default_rng(seed_value)
        chat_before = len(env.chat.get(env.step_count, []))
        env.step({tag: {"action": "noop", "messages": {"broadcast": f"hello from {tag}"}, "params": {}} for tag, _, _ in BEINGS})
        said = [line.split(":")[0] for line in env.chat[env.step_count - 1][chat_before:]]
        orders.add(tuple(said))
    assert len(orders) > 1


def test_the_lifespan_sentence_appears_only_when_life_is_limited(tmp_path):
    limited = social_world(tmp_path)
    assert texts.INSTRUCTIONS_LIFESPAN in library_of(limited).instructions(limited)
    unlimited = OpenGridWorld(grid_size=5, vision_radius=2, log_path=tmp_path / "u", **{**WORLD, "lifespan": -1})
    unlimited.attach(Library(Options()))
    assert texts.INSTRUCTIONS_LIFESPAN not in library_of(unlimited).instructions(unlimited)


def test_a_grid_world_hosts_the_library(tmp_path):
    env = OpenGridWorld(grid_size=5, vision_radius=2, log_path=tmp_path, **WORLD)
    env.attach(Library(Options()))
    env.add_agent("a", "Alice", "no_traits", position=(0, 0))
    env.add_agent("b", "Bob", "no_traits", position=(4, 4))
    env.restart_env(seed=1, agent_poses={"a": (0, 0), "b": (4, 4)})
    seed(env, (0, 0), "note")
    assert outcome(env, "a", "deposit_artifact", name="note") == texts.DEPOSITED.format(name="note")
    assert "note" not in env.pos_artifacts[(0, 0)]
    assert outcome(env, "b", "retrieve_artifact", name="note") == texts.RETRIEVED.format(name="note")
    assert "note" in env.pos_artifacts[(4, 4)]
