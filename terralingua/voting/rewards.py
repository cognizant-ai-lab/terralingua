"""Reward strategies: how a vote outcome's reward becomes agent energy.

A strategy is driven by a :class:`~terralingua.voting.manager.VotingManager` in the
post-step. ``on_outcome`` runs once per elected outcome (before any reward is
known); ``on_reward`` runs for each reward value the external world returned.
Both apply energy through ``env.inject_resource(amount, coefficient, target)``.

- :class:`DirectRewards` credits the raw reward to the single actor (independent
  mode): each agent keeps the outcome of its own call.
- :class:`DiasRewards` splits the reward equally across the contributing
  coalition and shares the action cost (the collective-action path),
  normalizing the reward against per-topic history so volatile rewards don't
  swamp energy.
"""

from collections import defaultdict, deque


class RewardStrategy:
    """Base strategy: distribute an outcome's reward as energy via *env*."""

    def get_state_ckpt(self) -> dict:
        return {}

    def set_state_ckpt(self, state: dict) -> None:
        """Stateless strategies have nothing to restore."""

    def on_outcome(self, outcome, topic, env) -> None:
        """One-time settlement for an elected outcome, before any reward."""

    def on_reward(self, outcome, topic, reward: float, env) -> None:
        """Settle one reward value returned for *outcome*."""


class DirectRewards(RewardStrategy):
    """Credit the raw reward to the representative (the sole actor)."""

    def __init__(self, coefficient: float):
        self.coefficient = float(coefficient)

    def on_reward(self, outcome, topic, reward: float, env) -> None:
        if self.coefficient == 0.0:
            return
        env.inject_resource(
            amount=float(reward),
            coefficient=self.coefficient,
            target=outcome.representative,
        )


class DiasRewards(RewardStrategy):
    """DIAS coalition settlement: share the action cost equally across
    contributors and split a history-normalized reward among them, penalizing
    misaligned voters."""

    def __init__(self, cost: float, coefficient: float):
        self.cost = float(cost)
        self.coefficient = float(coefficient)
        self._reward_history: dict = defaultdict(lambda: deque(maxlen=1000))

    def on_outcome(self, outcome, topic, env) -> None:
        contributors = outcome.contributors
        if self.cost <= 0 or len(contributors) <= 1:
            return
        # The action cost (charged to the representative by the env) is split
        # equally across the coalition: reimburse the rep, debit everyone else.
        share = self.cost / len(contributors)
        env.inject_resource(
            amount=self.cost - share, coefficient=1.0, target=outcome.representative
        )
        for tag in contributors:
            if tag != outcome.representative:
                env.inject_resource(amount=-share, coefficient=1.0, target=tag)

    def on_reward(self, outcome, topic, reward: float, env) -> None:
        if self.coefficient == 0.0:
            return
        impact = self._normalized_impact(topic, reward)
        transfer_total = impact * (self.cost or 1.0) * (self.coefficient / 0.1)

        if outcome.contributors:
            share = transfer_total / len(outcome.contributors)
            for tag in outcome.contributors:
                env.inject_resource(amount=share, coefficient=1.0, target=tag)
        if outcome.misaligned:
            penalty = transfer_total / len(outcome.misaligned)
            for tag in outcome.misaligned:
                env.inject_resource(amount=-penalty, coefficient=1.0, target=tag)

    def get_state_ckpt(self) -> dict:
        # A list of pairs preserves integer topics across JSON checkpoints.
        return {"history": [[topic, list(values)] for topic, values in self._reward_history.items()]}

    def set_state_ckpt(self, state: dict) -> None:
        self._reward_history = defaultdict(lambda: deque(maxlen=1000))
        for topic, values in state.get("history", []):
            self._reward_history[topic] = deque(values, maxlen=1000)

    def _normalized_impact(self, topic, reward: float) -> float:
        """Approximate DIAS energy impact from recent reward history for *topic*:
        normalize against prior history (min and a high-percentile max), then
        append the current reward — mirroring palife's calculate_energy_impact."""
        history = self._reward_history[topic]
        impact = 0.0
        if history:
            lmin = min(history)
            sorted_hist = sorted(history)
            target_index = max(1, len(sorted_hist) - int((len(sorted_hist) * 10) / 100))
            fitness_range = (sorted_hist[target_index - 1] - lmin) or 1.0
            impact = (reward - lmin) / fitness_range
        history.append(float(reward))
        return float(impact)
