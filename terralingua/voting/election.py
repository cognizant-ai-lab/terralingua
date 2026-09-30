"""Pure plurality election over opaque string choices.

The reusable core of the voting layer — it votes over abstract choices and knows
nothing about external servers, actions, or any benchmark. Callers map their
domain objects to choice strings, build :class:`Ballot` objects, and call
:func:`elect`.

Ties break RANDOMLY (seeded by the run's global RNG). A lexicographic
tie-break would punish late-alphabet actions: an action that always loses ties
by spelling is one the agents learn to avoid for the wrong reason.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field


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


def elect(ballots: list[Ballot]) -> VoteOutcome | None:
    """Elect the most-supported choice, or None if there are no ballots.

    The winner has the most distinct supporters (ties broken by choice string);
    the representative is its lowest-sorted supporter; contributors back the
    winner and the rest are misaligned.
    """
    if not ballots:
        return None

    supporters: dict[str, set[str]] = defaultdict(set)
    for b in ballots:
        supporters[b.choice].add(b.voter)

    top = max(len(vs) for vs in supporters.values())
    winning_choice = random.choice(
        sorted(c for c, vs in supporters.items() if len(vs) == top)
    )
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
