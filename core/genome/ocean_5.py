import copy
import random
from dataclasses import asdict, dataclass, field, fields

from core.genome.base_genome import Genome as BaseGenome


@dataclass
class Genome(BaseGenome):
    """This genome is used to define the main personality traits of the agents.
    It is based on the HEXACO + Interpersonal Circumplex psychology theory for the personality traits.
    """

    honesty: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = calculating, status-seeking; 1 = sincere, modest, fair-minded.",
            "trait_type": "personality",
        },
    )
    neuroticism: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = calm, resilient; 1 = sensitive, cautious, easily worried.",
            "trait_type": "personality",
        },
    )
    extraversion: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = quiet, reserved; 1 = sociable, energetic, seeks stimulation.",
            "trait_type": "personality",
        },
    )
    agreeableness: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = tough-minded, critical, aggressive; 1 = forgiving, patient, conflict-averse.",
            "trait_type": "personality",
        },
    )
    conscientiousness: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = spontaneous, disorganised; 1 = diligent, disciplined, orderly.",
            "trait_type": "personality",
        },
    )
    openness: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = conventional, prefers routine; 1 = curious, imaginative, variety-seeking.",
            "trait_type": "personality",
        },
    )
    dominance: float = field(
        default=0.0,
        metadata={
            "range": (-1.0, 1.0),
            "descr": "-1 = submissive, accommodating; 1 = assertive, controlling, leads interactions.",
            "trait_type": "personality",
        },
    )
    fertility: float = field(
        default=0.8,
        metadata={
            "range": (0.5, 1.0),
            "descr": "0 = no interest in reproduction; 1 = extremely high desire to reproduce",
            "trait_type": "physical",
        },
    )

    def __post_init__(self):
        """Check value ranges"""
        for f in fields(self):
            lo, hi = f.metadata["range"]
            val = getattr(self, f.name)
            if not (lo <= val <= hi):
                raise ValueError(f"{f.name}={val} outside {lo}-{hi}")

    @classmethod
    def random(cls):
        """Generates completely random genome"""
        kwargs = {}
        for f in fields(cls):
            lo, hi = f.metadata["range"]
            is_int = f.metadata.get("int", False)
            if is_int:
                kwargs[f.name] = random.randint(int(lo), int(hi))
            else:
                kwargs[f.name] = random.uniform(lo, hi)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, float | int]:
        """Return the genome as a plain Python dict."""
        # Option A: use the standard helper
        return asdict(self)

    def from_dict(self, data: dict[str, float | int]) -> "Genome":
        """Create a Genome instance from a dictionary."""
        return Genome(**data)

    def as_string(self) -> str:
        """Return the genome as a multiline string including value ranges."""
        personality_lines = []
        physical_lines = []
        for f in fields(self):
            val = getattr(self, f.name)
            descr = f.metadata["descr"]
            is_int = f.metadata.get("int", False)
            trait_type = f.metadata.get("trait_type", "personality")
            line = (
                f"  {f.name} value: {val:.3f}  ({descr})"
                if not is_int
                else f"  {f.name} value: {val}  ({descr})"
            )
            if trait_type == "personality":
                personality_lines.append(line)
            elif trait_type == "physical":
                physical_lines.append(line)
            else:
                raise ValueError(f"Trait type {trait_type} for {f.name} not recognized")

        personality_lines = "\n".join(personality_lines)
        physical_lines = "\n".join(physical_lines)

        string = f"=== Your Traits ===\nPersonality traits \n {personality_lines} \n\n Physical traits \n {physical_lines}"
        return string

    @classmethod
    def random_jitter(cls, **kwargs):
        """Start from defaults (optionally overridden) and jitter them."""
        return cls(**kwargs).mutate()

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ---------- mutation ---------- #
    def mutate(self, rate: float = 0.5, sigma: float = 0.3) -> "Genome":
        """Return a slightly mutated copy."""
        child = copy.deepcopy(self)

        for f in fields(child):  # iterate over dataclass fields
            if random.random() >= rate:
                continue  # gene stays the same

            lo, hi = f.metadata["range"]
            is_int = f.metadata.get("int", False)
            val = getattr(child, f.name)

            if is_int:
                val += random.choice([-1, 0, 1])
                val = int(self.clamp(val, lo, hi))
            else:
                val += random.gauss(0, sigma)
                val = self.clamp(val, lo, hi)

            setattr(child, f.name, val)

        return child


if __name__ == "__main__":
    gen = Genome()
    print(gen.as_dict())
    print(gen.as_string())
    print()
