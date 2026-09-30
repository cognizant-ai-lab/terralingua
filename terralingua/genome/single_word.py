from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from terralingua.genome.base_genome import Genome as BaseGenome


# ------------------ Embedding backend ------------------ #
class HFEmbedder:
    """Tiny Hugging Face embedder (BERT-tiny). L2-normalized mean-pooled embeddings."""

    def __init__(self, model_name: str = "prajjwal1/bert-tiny", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        out_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            last = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vecs = mean.detach().cpu().numpy().astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            out_vecs.append(vecs)
        return np.vstack(out_vecs)


# ------------------ Semantic space (max freedom) ------------------ #
class SemanticSpace:
    """Tokenizer-wide alphabetic vocabulary and their embeddings (nearest-neighbor decode)."""

    def __init__(self, embedder: HFEmbedder, min_len=3, max_len=16, lowercase=True):
        self.E = embedder
        raw = list(self.E.tokenizer.get_vocab().keys())

        if lowercase:
            seen, vocab = set(), []
            for t in raw:
                tl = t.lower()
                if tl.isalpha() and (min_len <= len(tl) <= max_len) and tl not in seen:
                    seen.add(tl)
                    vocab.append(tl)
        else:
            vocab = [t for t in raw if t.isalpha() and (min_len <= len(t) <= max_len)]
        if not vocab:
            raise RuntimeError("Tokenizer produced an empty alphabetic vocabulary.")

        self.vocab: List[str] = vocab
        self.M: np.ndarray = self.E.embed(
            self.vocab, batch_size=1024
        )  # (|V|, d), L2-normalized
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, word: str) -> np.ndarray:
        return self.E.embed([word.lower()])[0]

    def decode_nearest(self, vec: np.ndarray) -> Tuple[str, float]:
        sims = self.M @ vec
        i = int(np.argmax(sims))
        return self.vocab[i], float(sims[i])

    def decode_sample(
        self, vec: np.ndarray, temperature: float = 20.0
    ) -> Tuple[str, float]:
        sims = self.M @ vec
        logits = sims * temperature
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        i = int(np.random.choice(len(self.vocab), p=p))
        return self.vocab[i], float(sims[i])


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n


# ------------------ Word Genome ------------------ #
@dataclass
class Genome(BaseGenome):
    """
    A single-word semantic genotype/phenotype.
    Genotype lives on the unit sphere; phenotype is the nearest alphabetic token.
    """

    word: str = "neutral"

    # class-level cached model/space (built once)
    _emb: ClassVar[HFEmbedder | None] = None
    _space: ClassVar[SemanticSpace | None] = None

    # ---- setup / factory ---- #
    @classmethod
    def _ensure_space(
        cls,
        model_name="prajjwal1/bert-tiny",
        device="cpu",
        min_len=3,
        max_len=16,
        lowercase=True,
    ):
        if cls._emb is None:
            cls._emb = HFEmbedder(model_name=model_name, device=device)
        if cls._space is None:
            cls._space = SemanticSpace(
                cls._emb, min_len=min_len, max_len=max_len, lowercase=lowercase
            )

    @classmethod
    def random(cls) -> "Genome":
        """Pick a random alphabetic token from the tokenizer-wide vocab."""
        cls._ensure_space()
        w = random.choice(cls._space.vocab)  # type: ignore
        return cls(word=w)

    # ---- API parity helpers ---- #
    def as_dict(self) -> dict[str, str]:
        return {"word": self.word}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Genome":
        return cls(word=data["word"])

    def as_string(self) -> str:
        string = f"=== One-word Personality description ===\n  {self.word}"
        return string

    # ---- mutation ---- #
    def mutate(self, rate: float = 0.5, sigma: float = 0.12) -> "Genome":
        """
        Return a mutated copy. With probability 1-rate, returns an exact copy.
        - rate: mutate or not
        - sigma: semantic step size (0.02–0.1 small, 0.1–0.25 moderate, 0.3+ large)
        """
        self._ensure_space()

        child = copy.deepcopy(self)
        if random.random() >= rate:
            return child

        g = self._space.encode(self.word)  # type: ignore # unit vector
        step = np.random.normal(0.0, 1.0, size=g.shape).astype(np.float32)
        g2 = _l2norm(g + sigma * step)
        new_word, _ = self._space.decode_nearest(g2)  # type: ignore

        child.word = new_word
        return child

    # ---- distance ---- #
    def semantic_distance(self, other: "Genome") -> float:
        """Angular distance (radians) between self.word and other.word under the embedder."""
        self._ensure_space()
        v1, v2 = self._space.encode(self.word), self._space.encode(other.word)  # type: ignore
        cos = float(np.clip(v1 @ v2, -1.0, 1.0))
        return float(math.acos(cos))


# ------------------ Demo ------------------ #
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    g = Genome.random()
    print("Initial:", g.as_string())
    for s in [0.05, 0.12, 0.25, 0.4]:
        child = g.mutate(rate=1.0, sigma=s)
        print(f"σ={s:>4}: {g.word} → {child.word}")
