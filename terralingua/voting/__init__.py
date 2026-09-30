"""Reusable voting layer: pure election, reward strategies, and managers.

Generic over any setting where agents propose actions and a group outcome must
be chosen and rewarded — not tied to external servers or any benchmark.
"""

from terralingua.voting.election import Ballot, VoteOutcome, elect
from terralingua.voting.manager import (
    IndependentVotingManager,
    UnanimousVotingManager,
    Validation,
    VotingManager,
    action_key,
)
from terralingua.voting.rewards import DiasRewards, DirectRewards, RewardStrategy

__all__ = [
    "Ballot",
    "VoteOutcome",
    "elect",
    "action_key",
    "RewardStrategy",
    "DiasRewards",
    "DirectRewards",
    "VotingManager",
    "IndependentVotingManager",
    "UnanimousVotingManager",
    "Validation",
]
