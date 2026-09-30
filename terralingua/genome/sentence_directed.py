from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple

import litellm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from terralingua.genome.base_genome import Genome as BaseGenome

log = logging.getLogger(__name__)


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
        # (|V|, d), L2-normalized
        self.M: np.ndarray = self.E.embed(self.vocab, batch_size=1024)
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


# ------------------ Sentence-directed Genome ------------------ #
@dataclass
class Genome(BaseGenome):
    """
    A free-form natural-language sentence written by the parent agent at
    spawn time.  When the parent passes an empty string, the fallback
    mutation applies a word-by-word semantic walk: only alphabetic tokens of
    3+ characters are eligible, so grammar words are preserved.
    """

    genome_type = "sentence_directed"
    sentence: str = ""

    # class-level cached model/space (built once per process)
    _emb: ClassVar[HFEmbedder | None] = None
    _space: ClassVar[SemanticSpace | None] = None
    _crossover_model: ClassVar[str] = "anthropic/claude-haiku-4-5"
    _crossover_retries: ClassVar[int] = 10

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
        return cls(sentence="")

    # ---- serialisation ---- #
    def as_dict(self) -> dict:
        return {"sentence": self.sentence}

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(sentence=data.get("sentence", ""))

    def as_string(self) -> str:
        if not self.sentence:
            return "=== Personality sentence ===\n  (none)"
        return f"=== Personality sentence ===\n  {self.sentence}"

    # ---- mutation (fallback path when parent passes empty string) ---- #
    def mutate(self, rate: float = 0.5, sigma: float = 0.12) -> "Genome":
        """
        Return a mutated copy via word-by-word semantic substitution.
        Only alphabetic tokens of 3+ characters are mutated; short words and
        punctuation are preserved so the grammar stays intact.
        rate=0 returns an exact copy.
        """
        child = copy.deepcopy(self)
        if rate == 0.0 or not self.sentence.strip():
            return child

        self._ensure_space()
        tokens = self.sentence.split()
        new_tokens = []
        for token in tokens:
            clean = token.strip(".,!?;:'\"()[]{}")
            if clean.isalpha() and len(clean) >= 3 and random.random() < rate:
                g = self._space.encode(clean.lower())  # type: ignore
                step = np.random.normal(0.0, 1.0, size=g.shape).astype(np.float32)
                g2 = _l2norm(g + sigma * step)
                new_word, _ = self._space.decode_nearest(g2)  # type: ignore
                new_tokens.append(new_word)
            else:
                new_tokens.append(token)
        child.sentence = " ".join(new_tokens)
        return child

    def crossover(self, other: "Genome") -> "Genome":
        """LLM-blended crossover: ask an LLM to blend both parent sentences.

        Retries up to _crossover_retries times before falling back to
        word-level uniform crossover. Also falls back if either sentence is empty.
        """
        sentence_a = self.sentence.strip()
        sentence_b = other.sentence.strip()
        if not sentence_a or not sentence_b:
            return self._word_crossover(other)
        messages = [
            {
                "role": "system",
                "content": (
                    "You blend two personality descriptions into one offspring personality. "
                    "Reply with a single short sentence (max 15 words). "
                    "No quotes, no explanation, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Parent A: {sentence_a}\nParent B: {sentence_b}\nOffspring:",
            },
        ]
        for attempt in range(self._crossover_retries):
            try:
                resp = litellm.completion(
                    model=self._crossover_model,
                    messages=messages,
                    max_tokens=60,
                )
                sentence = resp.choices[0].message.content.strip().strip("\"'")  # type: ignore
                if sentence:
                    return Genome(sentence=sentence)
            except Exception as exc:
                log.warning(
                    "SentenceDirectedGenome.crossover attempt %d/%d failed: %s",
                    attempt + 1,
                    self._crossover_retries,
                    exc,
                )
        log.warning(
            "SentenceDirectedGenome.crossover: all retries exhausted, using word-level fallback."
        )
        return self._word_crossover(other)

    def _word_crossover(self, other: "Genome") -> "Genome":
        """Fallback: word-level uniform crossover (no LLM required)."""
        tokens_a = self.sentence.split()
        tokens_b = other.sentence.split()
        length = max(len(tokens_a), len(tokens_b))
        tokens_a += [""] * (length - len(tokens_a))
        tokens_b += [""] * (length - len(tokens_b))
        child_tokens = [
            a if random.random() < 0.5 else b for a, b in zip(tokens_a, tokens_b)
        ]
        return Genome(sentence=" ".join(t for t in child_tokens if t))


# ------------------ Demo ------------------ #
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    g = Genome(sentence="you are an idealist that wants to change the world")
    print("Initial:", g.as_string())
    for s in [0.05, 0.12, 0.25, 0.4]:
        child = g.mutate(rate=1.0, sigma=s)
        print(f"σ={s:>4}: {child.sentence}")
