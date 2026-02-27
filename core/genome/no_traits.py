import copy
from dataclasses import asdict, dataclass, field, fields

from core.genome.base_genome import Genome as BaseGenome


@dataclass
class Genome(BaseGenome):
    """This genome has no traits. It is used for agents that do not require any personality traits."""

    def as_dict(self) -> dict[str, float]:
        return {}

    def from_dict(self, data: dict[str, float]) -> "Genome":
        return copy.deepcopy(self)

    def as_string(self) -> str:
        return ""

    def mutate(self, **kwargs) -> "Genome":
        return copy.deepcopy(self)

    @classmethod
    def random(cls) -> "Genome":
        """Generates a genome with no traits."""
        return cls()
