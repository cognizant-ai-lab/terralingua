"""Tests for the reusable voting layer (terralingua.voting).

Covers the pure election kernel, the reward strategies (equal-share DIAS and
direct), and the managers that aggregate agents' external actions — the behavior
that previously lived in the in-runner unanimous coordinator.
"""

import json
import random

import numpy as np
import pytest

from terralingua.voting.election import Ballot, VoteOutcome, elect
from terralingua.voting.manager import (
    IndependentVotingManager,
    UnanimousVotingManager,
    action_key,
)
from terralingua.voting.rewards import DiasRewards, DirectRewards


class _StubEnv:
    def __init__(self):
        self.injections = []  # list of (amount, coefficient, target)

    def inject_resource(self, amount, coefficient, target):
        self.injections.append((amount, coefficient, target))


def _ext_action(args, message=""):
    return {"action": "world_step", "params": dict(args), "message": message}


# ---- election ---------------------------------------------------------------

def test_elect_plurality_winner_rep_and_classes():
    out = elect([Ballot("a0", "plant"), Ballot("a1", "plant"), Ballot("a2", "wait")])
    assert out.winning_choice == "plant"
    assert out.representative == "a0"  # lowest-sorted supporter of the winner
    assert out.contributors == {"a0", "a1"}
    assert out.misaligned == {"a2"}


def test_elect_tie_breaks_randomly_among_tied():
    """Ties break by seeded randomness, not spelling — the lexicographic
    rule punished late-alphabet actions (an action that always loses ties by spelling)."""
    winners = set()
    for seed in range(20):
        out = elect([Ballot("a0", "b"), Ballot("a1", "a")], np.random.default_rng(seed))
        assert out.winning_choice in ("a", "b")
        assert out.representative == ("a1" if out.winning_choice == "a" else "a0")
        winners.add(out.winning_choice)
    assert winners == {"a", "b"}  # both sides of the tie can win
    # The same generator state gives the same winner: seeded runs repeat.
    tie = [Ballot("a0", "b"), Ballot("a1", "a")]
    assert elect(tie, np.random.default_rng(3)).winning_choice == elect(tie, np.random.default_rng(3)).winning_choice
    # A clear majority is never randomized away.
    out = elect([Ballot("a0", "b"), Ballot("a1", "a"), Ballot("a2", "a")], np.random.default_rng(0))
    assert out.winning_choice == "a"


def test_elect_empty_returns_none():
    assert elect([]) is None


# ---- reward strategies ------------------------------------------------------

def test_dias_cost_shared_equally():
    env = _StubEnv()
    out = VoteOutcome("plant", "a0", contributors={"a0", "a1"})
    DiasRewards(cost=4.0, coefficient=0.0).on_outcome(out, 0, env)
    by = {t: a for a, c, t in env.injections}
    assert by["a0"] == pytest.approx(2.0)   # 4 - 4/2 reimbursed
    assert by["a1"] == pytest.approx(-2.0)  # charged its 4/2 share


def test_dias_cost_skipped_for_solo():
    env = _StubEnv()
    DiasRewards(cost=4.0, coefficient=0.0).on_outcome(VoteOutcome.solo("a0"), 0, env)
    assert env.injections == []


def test_dias_reward_zero_coeff_noop():
    env = _StubEnv()
    out = VoteOutcome("p", "a0", contributors={"a0"})
    DiasRewards(cost=5.0, coefficient=0.0).on_reward(out, 0, 10.0, env)
    assert env.injections == []


def test_direct_reward_zero_coeff_noop():
    env = _StubEnv()
    DirectRewards(coefficient=0.0).on_reward(VoteOutcome.solo("a0"), "a0", 10.0, env)
    assert env.injections == []


# ---- unanimous manager ------------------------------------------------------

def test_unanimous_groups_by_slot_and_suppresses_losers():
    mgr = UnanimousVotingManager(rewards=DiasRewards(5.0, 0.2), vote_key="slot")
    actions = {
        "a0": _ext_action({"slot": 0, "action": "plant"}),
        "a1": _ext_action({"slot": 0, "action": "plant"}),
        "a2": _ext_action({"slot": 1, "action": "wait"}),
    }
    requests = {tag: dict(a["params"]) for tag, a in actions.items()}
    no_op = {"action": "noop", "message": "", "params": {}}
    val = mgr.validate(actions, requests, no_op)

    assert mgr.active_topics() == {0, 1}
    assert set(val.to_dispatch) == {"a0", "a2"}        # one representative per slot
    assert val.broadcasts["a0"] == {"a1"}              # peer shares the rep's response
    assert actions["a0"]["action"] == "world_step"  # rep keeps the external action
    assert actions["a1"]["action"] == "noop"            # loser suppressed to no-op
    assert actions["a1"]["params"] == {}
    assert actions["a1"]["default_reason"] == "election_override"


def test_unanimous_settle_shares_reward():
    dias = DiasRewards(cost=0.0, coefficient=0.2)
    dias._reward_history[0].extend([0.0, 20.0])
    mgr = UnanimousVotingManager(rewards=dias, vote_key="slot")
    actions = {
        "a0": _ext_action({"slot": 0, "action": "plant"}),
        "a1": _ext_action({"slot": 0, "action": "plant"}),
        "a2": _ext_action({"slot": 0, "action": "wait"}),
    }
    requests = {tag: dict(a["params"]) for tag, a in actions.items()}
    mgr.validate(actions, requests, {"action": "noop", "params": {}})

    obs = {"a0": {"external_response": [json.dumps({"reward": 10})]}}
    env = _StubEnv()
    mgr.settle(obs, env)
    by = {t: a for a, c, t in env.injections}
    assert by["a0"] > 0 and by["a1"] > 0  # contributors rewarded
    assert by["a0"] == pytest.approx(by["a1"])  # equal split
    assert by["a2"] < 0                   # misaligned penalized


# ---- independent manager ----------------------------------------------------

def test_independent_dispatches_all_no_broadcast():
    mgr = IndependentVotingManager(rewards=DirectRewards(0.2))
    requests = {"a0": {"x": 1}, "a1": {"x": 2}}
    val = mgr.validate({}, requests, {"action": "noop", "params": {}})
    assert val.to_dispatch == requests
    assert val.broadcasts == {}
    assert mgr.active_topics() == set()


def test_independent_credits_each_caller():
    mgr = IndependentVotingManager(rewards=DirectRewards(0.2))
    mgr.validate({}, {"a0": {"x": 1}, "a1": {"x": 2}}, {"action": "noop", "params": {}})
    obs = {
        "a0": {"external_response": [json.dumps({"reward": 10})]},
        "a1": {"external_response": [json.dumps({"reward": 5})]},
    }
    env = _StubEnv()
    mgr.settle(obs, env)
    by = {t: a for a, c, t in env.injections}
    assert by["a0"] == pytest.approx(10.0)
    assert by["a1"] == pytest.approx(5.0)


# ── ballot canonicalization: same meaning, one choice ────────────────────────


def test_action_key_groups_same_meaning_ballots():
    # Reordered and re-spaced call-syntax strings are one choice.
    a = action_key({"action": "place(x=1, y=2)", "slot": 0})
    b = action_key({"action": "place(y=2,x=1)", "slot": "0"})
    assert a == b
    # Numeric spellings are one choice.
    assert action_key({"slot": "1.0"}) == action_key({"slot": 1})
    # Genuinely different actions never merge.
    assert action_key({"action": "place(x=1, y=2)"}) != action_key(
        {"action": "place(x=1, y=3)"}
    )
    assert action_key({"action": "wait()"}) != action_key(
        {"action": "remove(x=1)"}
    )
    # Unparseable strings fall back to trimmed-text identity.
    assert action_key({"note": " hello "}) == action_key({"note": "hello"})
    assert action_key({"note": "hello"}) != action_key({"note": "goodbye"})


def test_unanimous_validation_exposes_vote_outcomes():
    manager = UnanimousVotingManager(rewards=None, vote_key="slot")
    actions = {
        "a": {"action": "world_step", "params": {"slot": "0", "action": "x()"}},
        "b": {"action": "world_step", "params": {"slot": "0", "action": "x()"}},
        "c": {"action": "world_step", "params": {"slot": "0", "action": "y()"}},
    }
    requests = {tag: dict(actions[tag]["params"]) for tag in actions}
    validation = manager.validate(actions, requests, {"action": "noop", "params": {}})
    assert set(validation.outcomes) == {"0"}
    outcome = validation.outcomes["0"]
    assert outcome.representative == "a"
    assert outcome.contributors == {"a", "b"}
    assert outcome.misaligned == {"c"}
