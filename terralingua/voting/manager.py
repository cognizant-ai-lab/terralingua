"""Voting managers: aggregate agents' external actions and reward the results.

A manager owns one external server's collective-action policy. Each step the
runner asks it to :meth:`validate` the agents' requests — deciding which actions
are actually enacted — and later to :meth:`settle` the rewards the server
returned. It delegates *how* rewards are computed to a
:class:`~terralingua.voting.rewards.RewardStrategy`, so the policy (vote vs. not) and
the payout (DIAS vs. direct) vary independently.

- :class:`IndependentVotingManager` — no contention: every request is enacted
  and each agent keeps its own reward.
- :class:`UnanimousVotingManager` — agents targeting the same topic must agree;
  one representative's action is enacted per topic and the reward is shared.

The manager never talks to the MCP server itself: :meth:`validate` returns the
actions to dispatch and the runner performs the I/O.
"""

import json
from dataclasses import dataclass, field

from terralingua.voting.canonical import canonical_value, structural_parse
from terralingua.voting.election import Ballot, VoteOutcome, elect


def canonical_ballot_value(value):
    """Spelling-independent form of one arg value: numbers normalized,
    structured strings compared by their parsed fields."""
    if isinstance(value, str):
        fields = structural_parse(value)
        if fields is not None:
            return {k: canonical_ballot_value(v) for k, v in fields.items()}
        return canonical_value(value.strip())
    if isinstance(value, dict):
        return {k: canonical_ballot_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_ballot_value(v) for v in value]
    return canonical_value(value)


def action_key(args: dict) -> str:
    """Stable identity of an action's args: same-meaning proposals count as
    ONE choice even when spelled differently (arg order, spacing, "0" vs 0).
    Deterministic, no LLM; the key only groups ballots — the representative
    still executes its own original args verbatim."""
    return json.dumps(
        canonical_ballot_value(args), sort_keys=True, separators=(",", ":")
    )


@dataclass
class Validation:
    """Result of :meth:`VotingManager.validate` for one server this step."""

    # agent_tag → args to actually send to the external server.
    to_dispatch: dict = field(default_factory=dict)
    # representative tag → coalition peers that share its response/status.
    broadcasts: dict = field(default_factory=dict)
    # topic → VoteOutcome (unanimous mode only), so the caller can attribute
    # the dispatched call to its coalition and record overridden dissenters.
    outcomes: dict = field(default_factory=dict)


class VotingManager:
    def __init__(self, rewards):
        self.rewards = rewards

    def validate(
        self, actions: dict, requests: dict, no_op_action: dict
    ) -> Validation:
        raise NotImplementedError

    def settle(self, obs: dict, env) -> None:
        raise NotImplementedError

    def active_topics(self) -> set:
        """Topics acted on this step (for idle tracking). Empty by default."""
        return set()

    @staticmethod
    def _rewards_from(obs: dict, tag: str) -> list:
        """Reward values parsed from an agent's external responses this step."""
        rewards = []
        seen = set()
        for resp_str in obs.get(tag, {}).get("external_response", []):
            try:
                result = json.loads(resp_str)
                if not isinstance(result, dict) or result.get("error") or result.get("external_failure"):
                    continue
                identity = json.dumps(result, sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                reward = result.get("reward")
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                continue
            if reward is not None:
                try:
                    rewards.append(float(reward))
                except (TypeError, ValueError):
                    pass
        return rewards


class IndependentVotingManager(VotingManager):
    """Each caller acts on its own and keeps its own reward."""

    def __init__(self, rewards):
        super().__init__(rewards)
        self._callers: dict = {}

    def validate(
        self, actions: dict, requests: dict, no_op_action: dict
    ) -> Validation:
        self._callers = dict(requests)
        return Validation(to_dispatch=dict(requests))

    def settle(self, obs: dict, env) -> None:
        for tag in self._callers:
            outcome = VoteOutcome.solo(tag)
            self.rewards.on_outcome(outcome, tag, env)
            for reward in self._rewards_from(obs, tag):
                self.rewards.on_reward(outcome, tag, reward, env)


class UnanimousVotingManager(VotingManager):
    """Agents targeting the same topic must agree; one representative acts per
    topic and the reward is shared across the coalition.

    ``vote_key`` is the request-arg field whose value groups the step's requests
    into one election each (e.g. a world slot index). Empty => one global
    election over all requests."""

    def __init__(self, rewards, vote_key: str = ""):
        super().__init__(rewards)
        self.vote_key = vote_key
        self._outcomes: dict = {}

    def validate(
        self, actions: dict, requests: dict, no_op_action: dict
    ) -> Validation:
        self._outcomes = {}
        groups: dict = {}
        for tag, args in requests.items():
            topic = args.get(self.vote_key) if self.vote_key else None
            groups.setdefault(topic, {})[tag] = args

        to_dispatch: dict = {}
        broadcasts: dict = {}
        for topic, group in groups.items():
            outcome = elect([Ballot(tag, action_key(args)) for tag, args in group.items()])
            if outcome is None:
                continue
            self._outcomes[topic] = outcome
            rep = outcome.representative
            to_dispatch[rep] = group[rep]
            broadcasts[rep] = (outcome.contributors | outcome.misaligned) - {rep}

        # Only the elected representatives keep the external action; everyone else
        # who requested it becomes a no-op, so the env doesn't charge the
        # per-action cost again for an already-elected topic step. The no-op uses
        # the env's canonical do-nothing action (e.g. `noop` in social_graph,
        # `move(stay)` in grid worlds) so the override is valid for the world type.
        reps = set(to_dispatch)
        for tag in requests:
            if tag in reps:
                continue
            original = actions.get(tag, {})
            actions[tag] = {
                **no_op_action,
                "message": original.get("message", ""),
                "source": "default",
                "default_reason": "election_override",
            }
        return Validation(
            to_dispatch=to_dispatch,
            broadcasts=broadcasts,
            outcomes=dict(self._outcomes),
        )

    def settle(self, obs: dict, env) -> None:
        for topic, outcome in self._outcomes.items():
            self.rewards.on_outcome(outcome, topic, env)
            for reward in self._rewards_from(obs, outcome.representative):
                self.rewards.on_reward(outcome, topic, reward, env)

    def active_topics(self) -> set:
        return set(self._outcomes)
