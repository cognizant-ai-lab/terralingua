from __future__ import annotations

import copy
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


# ------------------ Semantic space ------------------ #
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
        self.M: np.ndarray = self.E.embed(self.vocab, batch_size=1024)  # (|V|, d), L2-normalized
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, word: str) -> np.ndarray:
        return self.E.embed([word.lower()])[0]

    def decode_nearest(self, vec: np.ndarray) -> Tuple[str, float]:
        sims = self.M @ vec
        i = int(np.argmax(sims))
        return self.vocab[i], float(sims[i])


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n


# ------------------ Sentence-mutate Genome ------------------ #
@dataclass
class Genome(BaseGenome):
    """
    A fixed-length sequence of K bare words, each living on the unit sphere.
    Mutation walks every word independently through semantic space.
    K is sampled uniformly from 1–20 at random() time and then fixed for the
    lifetime of that genome instance.
    """

    genome_type = "sentence_mutate"
    words: tuple = ("neutral",)

    # class-level cached model/space (built once per process)
    _emb: ClassVar[HFEmbedder | None] = None
    _space: ClassVar[SemanticSpace | None] = None

    # ---- setup ---- #
    @classmethod
    def _ensure_space(
        cls,
        model_name: str = "prajjwal1/bert-tiny",
        device: str = "cpu",
        min_len: int = 3,
        max_len: int = 16,
        lowercase: bool = True,
    ):
        if cls._emb is None:
            cls._emb = HFEmbedder(model_name=model_name, device=device)
        if cls._space is None:
            cls._space = SemanticSpace(
                cls._emb, min_len=min_len, max_len=max_len, lowercase=lowercase
            )

    # ---- factory ---- #
    @classmethod
    def random(cls) -> "Genome":
        """Sample a random number of words (1–20) and pick random vocab tokens."""
        cls._ensure_space()
        k = random.randint(1, 20)
        words = tuple(random.choice(cls._space.vocab) for _ in range(k))  # type: ignore
        return cls(words=words)

    # ---- serialisation ---- #
    def as_dict(self) -> dict:
        return {"words": list(self.words)}

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(words=tuple(data["words"]))

    def as_string(self) -> str:
        return f"=== Personality sentence ===\n  {' '.join(self.words)}"

    # ---- mutation ---- #
    def mutate(self, rate: float = 0.5, sigma: float = 0.12) -> "Genome":
        """
        Return a mutated copy. Each word is independently mutated with
        probability *rate* by taking a Gaussian step of size *sigma* in
        embedding space and decoding to the nearest vocab token.
        - rate:  probability that a given word mutates (0 = no mutation)
        - sigma: semantic step size (0.02–0.1 small, 0.1–0.25 moderate, 0.3+ large)
        """
        self._ensure_space()
        child = copy.deepcopy(self)
        new_words = list(child.words)
        for i, word in enumerate(self.words):
            if random.random() >= rate:
                continue
            g = self._space.encode(word)  # type: ignore
            step = np.random.normal(0.0, 1.0, size=g.shape).astype(np.float32)
            g2 = _l2norm(g + sigma * step)
            new_word, _ = self._space.decode_nearest(g2)  # type: ignore
            new_words[i] = new_word
        child.words = tuple(new_words)
        return child

    # ---- crossover ---- #
    def crossover(self, other: "Genome") -> "Genome":
        """Uniform crossover over word positions; child length = longer parent."""
        words_a = list(self.words)
        words_b = list(other.words)
        length = max(len(words_a), len(words_b))
        words_a += [random.choice(words_a) for _ in range(length - len(words_a))]
        words_b += [random.choice(words_b) for _ in range(length - len(words_b))]
        child_words = tuple(
            a if random.random() < 0.5 else b for a, b in zip(words_a, words_b)
        )
        return Genome(words=child_words)


# ------------------ Demo ------------------ #
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    g = Genome.random()
    print("Initial:", g.as_string())
    for s in [0.05, 0.12, 0.25, 0.4]:
        child = g.mutate(rate=1.0, sigma=s)
        print(f"σ={s:>4}: {g.words} → {child.words}")
