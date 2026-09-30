"""Pure plurality election over opaque string choices.

The reusable core of the voting layer — it votes over abstract choices and knows
nothing about external servers, actions, or any benchmark. Callers map their
domain objects to choice strings, build :class:`Ballot` objects, and call
:func:`elect`.

Ties break at random with the generator the caller passes in. The runner
passes the world's generator, which the run seed initialises and the checkpoint
saves and restores, so tied elections repeat across seeded runs and after a
resume. A lexicographic tie-break would punish late-alphabet actions: an action
that always loses ties by spelling is one the agents learn to avoid for the
wrong reason.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ballot:
    """One agent's vote: the voter and the choice string it emitted."""

    voter: str
    choice: str


@dataclass
class VoteOutcome:
    """Election result: the winning choice, the representative that executes it,
    and who agreed (contributors) vs. disagreed (misaligned)."""

    winning_choice: str
    representative: str
    contributors: set[str] = field(default_factory=set)
    misaligned: set[str] = field(default_factory=set)

    @classmethod
    def solo(cls, voter: str, choice: str = "") -> "VoteOutcome":
        """Trivial outcome for a single uncontested actor (independent mode)."""
        return cls(winning_choice=choice, representative=voter, contributors={voter})


def elect(ballots: list[Ballot], rng: np.random.Generator | None = None) -> VoteOutcome | None:
    """Elect the most-supported choice, or None if there are no ballots.

    The winner has the most distinct supporters. Ties break at random with
    `rng`; without one, a fresh unseeded generator is used. The representative
    is the winner's lowest-sorted supporter; contributors back the winner and
    the rest are misaligned.
    """
    if not ballots:
        return None

    supporters: dict[str, set[str]] = defaultdict(set)
    for b in ballots:
        supporters[b.choice].add(b.voter)

    top = max(len(vs) for vs in supporters.values())
    tied = sorted(c for c, vs in supporters.items() if len(vs) == top)
    generator = rng if rng is not None else np.random.default_rng()
    winning_choice = tied[int(generator.integers(len(tied)))]
    contributors = set(supporters[winning_choice])
    misaligned = {
        v for c, vs in supporters.items() if c != winning_choice for v in vs
    } - contributors

    return VoteOutcome(
        winning_choice=winning_choice,
        representative=min(contributors),
        contributors=contributors,
        misaligned=misaligned,
    )
