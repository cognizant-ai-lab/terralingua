import json
import logging
import pickle as pkl
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

import joblib
import numpy as np
import openai
import pandas as pd
import spacy
import torch
import zstandard as zstd
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from openai import NOT_GIVEN
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from terralingua.anthropologist.analysis_utils import get_last_ts, load_worldlog
from terralingua.utils import LOGS_DIR

load_dotenv(find_dotenv(usecwd=True), override=True)


# Metrics
# ================================
@dataclass
class Metric:
    name: str

    def _evaluate(self, text: str):
        raise NotImplementedError

    def compute(self, artifacts: "ExperimentArtifacts"):
        log.info("Calculating %s...", self.name)
        for tag in tqdm(artifacts.all_artifacts):
            val = self._evaluate(artifacts.all_artifacts[tag]["string"])
            artifacts.all_artifacts[tag][self.name] = val

        metric_by_ts = {"mean": [], "std": [], "max": [], "min": [], "median": []}
        values = []
        for ts, tags in tqdm(artifacts.get_artifact_by_creation().items()):
            try:
                values.extend([artifacts.all_artifacts[tag][self.name] for tag in tags])
                metric_by_ts["mean"].append(np.mean(values))
                metric_by_ts["std"].append(np.std(values))
                metric_by_ts["max"].append(np.max(values))
                metric_by_ts["min"].append(np.min(values))
                metric_by_ts["median"].append(np.median(values))
            except Exception as e:
                log.warning("Error at ts %s with tags %s: %s", ts, tags, e)
                metric_by_ts["mean"].append(0.0)
                metric_by_ts["std"].append(0.0)
                metric_by_ts["max"].append(0.0)
                metric_by_ts["min"].append(0.0)
                metric_by_ts["median"].append(0.0)
        artifacts.metrics[self.name] = metric_by_ts
        artifacts.save_metrics()
        log.info("%s calculation done.", self.name)


@dataclass
class LMSurprisal(Metric):
    """
    LM surprisal is the negative log-likelihood of the observed text under a language model, normalized by token count.
    It measures how unexpected the text is according to the model’s learned distribution.

    Low surprisal → the text is statistically predictable under the LM (linguistically simple, stereotypical phrasing).
        High surprisal → the LM assigns low probability to the actual sequence (rare constructions, complex syntax, unusual vocabulary).
    """

    name: str = "LMSurprisal"
    model_id: str = "gpt2-medium"
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return_perplexity: bool = False
    max_stride: int = 900  # < model max_len to allow overlap

    def __post_init__(self):
        log.info("Loading Tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        log.info("Loading LM...")
        self.lm = (
            AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device).eval()  # type: ignore
        )
        log.info("Done")

    @torch.no_grad()
    def _evaluate(self, text: str):
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        n_tokens = input_ids.size(1)
        if n_tokens == 0:
            return 0.0

        out_losses = []
        covered = 0

        max_len = getattr(self.lm.config, "n_positions", 1024)
        stride = min(self.max_stride, max_len - 1)
        start = 0
        while start < n_tokens:
            end = min(start + max_len, n_tokens)
            chunk = input_ids[:, start:end]  # [1, L]
            # build labels that predict next token; last is ignored
            labels = torch.full_like(chunk, -100)  # [1, L]
            labels[:, :-1] = chunk[:, 1:]  # no overlap issue
            # optional: attention mask if you ever pad chunks
            out = self.lm(chunk, labels=labels, use_cache=False)
            valid = (labels != -100).sum().item()
            out_losses.append(out.loss.item() * valid)
            covered += valid
            if end == n_tokens:
                break
            start = end - stride

        mean_nll = sum(out_losses) / max(1, covered)
        return (
            float(torch.exp(torch.tensor(mean_nll)))
            if self.return_perplexity
            else float(mean_nll)
        )


@dataclass
class CompressedSize(Metric):
    level: int = 5
    name: str = "CompressedSize"

    def __post_init__(self):
        self.cctx = zstd.ZstdCompressor(level=self.level)

    def _evaluate(self, text: str):
        raw = text.encode("utf-8")
        if not raw:
            return 0.0
        comp = self.cctx.compress(raw)
        # subtract fixed-frame overhead (~22–30 bytes)
        overhead = 24
        compression = max(1, len(comp) - overhead)
        return compression


@dataclass
class InverseCompressionRate(Metric):
    """
    Compression rate measures the redundancy and regularity of the text.
    It's a proxy for information density and unpredictability.

    Low NCL (≈0.2-0.5): text is highly regular, repetitive, or predictable.
        Examples: boilerplate, lists of similar items, simple syntactic structures.
    High NCL (≈0.7-1.0): text is more irregular, less compressible, structurally and lexically varied.
        Indicates higher intrinsic informational complexity.
    """

    level: int = 5
    name: str = "InverseCompressionRate"

    def __post_init__(self):
        self.cctx = zstd.ZstdCompressor(level=self.level)

    def _evaluate(self, text: str):
        raw = text.encode("utf-8")
        if not raw:
            return 0.0
        comp = self.cctx.compress(raw)
        # subtract fixed-frame overhead (~22–30 bytes)
        overhead = 24
        num = max(1, len(comp) - overhead)
        den = max(1, len(raw))
        return num / den


@dataclass
class SyntacticDepth(Metric):
    """
    Syntactic depth quantifies how structurally nested a sentence is.
    It measures the length of the longest dependency path from any token to the root of the dependency tree.

    High values indicate deeply nested constructions—relative clauses, embedded clauses,
    center-embedding, or heavy modifier stacks—typical of syntactically complex sentences.
    Low values correspond to flat, simple, paratactic structures.
    """

    name: str = "SyntacticDepth"
    model: str = "en_core_web_sm"
    metric: str = "mean_dep_depth"

    def __post_init__(self):
        self.avail_metrics = ["mean_dep_depth", "max_dep_depth", "avg_dep_distance"]
        assert self.metric in self.avail_metrics, (
            f"Metric {self.metric} not available. Choose one among: {self.avail_metrics}"
        )
        self.nlp = spacy.load(self.model, disable=[])
        # if speed-critical: disable ner, keep tagger+parser

    def _evaluate(self, text: str):
        doc = self.nlp(text)
        depths, dists = [], []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            # depth to ROOT
            d = 0
            cur = tok
            while cur.head != cur:
                d += 1
                cur = cur.head
            depths.append(d)
            # distance to head
            if tok.head != tok:
                dists.append(abs(tok.i - tok.head.i))
        if self.metric == "mean_dep_depth":
            return (sum(depths) / len(depths)) if depths else 0.0
        elif self.metric == "max_dep_depth":
            return max(depths) if depths else 0
        elif self.metric == "avg_dep_distance":
            return (sum(dists) / len(dists)) if dists else 0.0
        else:
            raise ValueError(
                f"Metric {self.metric} not available. Choose one among: {self.avail_metrics}"
            )


@dataclass
class LexicalSophistication(Metric):
    """
    Lexical sophistication measures how rare or informationally dense the vocabulary in a text is, relative to some reference distribution.
    It quantifies how much the text uses infrequent, specialized, or high-information words rather than common, high-frequency ones.
    """

    name: str = "LexicalSophistication"
    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    dataset_split: str = "train[:3%]"

    def __post_init__(self):
        self.save_path = LOGS_DIR / "vectorizer.joblib"
        self._TOKEN_RE = re.compile(r"(?u)\b\w+\b")

        if self.save_path.exists():
            log.info("Vectorized dataset found. Loading...")
            self.vect = joblib.load(self.save_path)
        else:
            log.info("Creating vectorized dataset...")
            # Fit the vectorizer on the given corpus
            self.vect = TfidfVectorizer(
                lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=5
            )
            log.info("Loading data corpus...")
            ds = load_dataset(
                self.dataset_name, self.dataset_config, split=self.dataset_split
            )
            corpus = [re.sub(r"\s+", " ", x["text"]) for x in ds if x["text"].strip()]  # type: ignore
            log.info("Fitting vectorizer...")
            self.vect.fit(corpus)
            log.info("Saving...")
            joblib.dump(self.vect, self.save_path)
            log.info("Done.")

        vocab = self.vect.vocabulary_
        idf = self.vect.idf_
        self.idf_map = {tok: idf[idx] for tok, idx in vocab.items()}
        # OOV handling: assign the max observed IDF (most “rare” seen in reference)
        self.oov_idf = float(idf.max())

    def _evaluate(self, text: str):
        toks = [t.lower() for t in self._TOKEN_RE.findall(text)]
        if not toks:
            return 0.0
        vals = [self.idf_map.get(t, self.oov_idf) for t in toks]
        return sum(vals) / len(vals)


class ExpansionMap:
    def __init__(self, dimension, epsilon, metric="euclidean"):
        """
        Initialize the ExpansionMap.

        Parameters:
        dimension (int): The dimension D of the space R^D.
        epsilon (float): The threshold distance to consider when adding new points.
        """
        self.device = torch.device("cpu")
        self.dimension = dimension

        # Initialize an empty tensor for storing points, and set it to the device
        self.leaders = torch.empty(0, dimension, device=self.device)
        self.epsilon = epsilon

        self.ROOT_ID = -1
        self.leader_ids: List[int] = []

        self.leader_tree = defaultdict(list)
        self.parent: Dict[int, int] = {}
        self.children: Dict[int, List[int]] = {self.ROOT_ID: []}

        self.point_ids: List[int] = []
        self.leader_by_ts: Dict[int, List[int]] = defaultdict(list)
        # Used to match leader_by_ts with leader index in self.leader and self.leader_ids
        self.leader_counter = 0
        self.metric = metric

    def as_dict(self) -> Dict:
        return {
            "leader_ids": self.leader_ids,
            "parent": self.parent,
            "children": self.children,
            "leader_tree": self.leader_tree,
            "point_ids": self.point_ids,
        }

    def from_dict(self, data: Dict) -> None:
        self.leader_ids = data["leader_ids"]
        self.parent = data["parent"]
        self.children = data["children"]
        self.point_ids = data["point_ids"]
        self.leader_tree = data["leader_tree"]

    def add_point(
        self,
        new_point: np.ndarray | torch.Tensor,
        point_id: int,
        timestamp: int,
    ) -> bool:
        """
        Attempt to add a new point to the map.

        Parameters:
        new_point (tensor): A tensor representing the new point (should be 1xD).
        id (int): The unique identifier for the new point.

        Returns:
        bool: True if the point was added, False if it was discarded.
        """
        # Ensure the new point is on the same device as the map points
        if point_id in self.point_ids:
            return False

        new_point = torch.Tensor(new_point).to(self.device).reshape(1, -1)
        self.point_ids.append(point_id)

        # if self.leaders.numel() == 0:
        #     self.leader_tree[self.ROOT_ID].append(point_id)
        #     self.leader_tree[point_id] = []
        #     self._add_leader(point_id, new_point, timestamp=timestamp)
        #     return True

        # Get all leaders from previous timestamps
        prev_leader_idx = []
        for ts in self.leader_by_ts:
            if ts < timestamp:
                prev_leader_idx.extend(self.leader_by_ts[ts])
        if prev_leader_idx:
            leaders = self.leaders[prev_leader_idx]

        # Add all artifacts at ts = 0
        else:
            self.leader_tree[self.ROOT_ID].append(point_id)
            self.leader_tree[point_id] = []
            self._add_leader(point_id, new_point, timestamp=timestamp)
            return True

        if len(new_point.shape) == 1:
            new_point = new_point.unsqueeze(0)

        if self.metric == "euclidean":
            distances = torch.norm(leaders - new_point, dim=1)
        elif self.metric == "cosine":
            distances = 1 - torch.cosine_similarity(leaders, new_point, dim=1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        dmin, argmin = torch.min(distances, dim=0)
        leader_id = self.leader_ids[prev_leader_idx[int(argmin.item())]]

        if float(dmin.item()) > self.epsilon:
            # Each new leader is added as child of the closest previous leader in the leader tree
            self.leader_tree[leader_id].append(point_id)
            self.leader_tree[point_id] = []
            self._add_leader(point_id, new_point, timestamp=timestamp)
            return True
        else:
            # Each new follower is added as child of the closest leader
            self.parent[point_id] = leader_id
            self.children.setdefault(leader_id, []).append(point_id)
            return False

    def _add_leader(self, point_id: int, x: torch.Tensor, timestamp: int) -> None:
        # Each new leader is added as child of the ROOT
        self.parent[point_id] = self.ROOT_ID
        self.children.setdefault(self.ROOT_ID, []).append(point_id)
        self.children.setdefault(point_id, [])

        self.leader_ids.append(point_id)
        if self.leaders.numel() == 0:
            self.leaders = x.clone()
        else:
            self.leaders = torch.cat([self.leaders, x], dim=0)

        self.leader_by_ts[timestamp].append(self.leader_counter)
        self.leader_counter += 1

    def follower_counts(self) -> Dict[int, int]:
        """Number of followers for each leader."""
        return {lid: len(self.children.get(lid, [])) for lid in self.leader_ids}

    def leader_counts(self):
        return len(self.leader_ids)

    def tree_summary(self) -> Dict:
        fc = self.follower_counts()
        return {
            "num_leaders": self.leader_counts(),
            "follower_counts": [fc[l] for l in self.leader_ids],
        }


# ================================


class ExperimentArtifacts:
    def __init__(
        self,
        exp_path: Path | str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int | None = None,
        save_path: Path | str | None = None,
        replace_numbers: bool = False,
        embed_names: bool = True,
        expansion_epsilon: float = 0.75,
        expansion_metric: str = "euclidean",
    ):
        exp_path = Path(exp_path)
        assert exp_path.exists(), f"Missing path: {self.exp_path}"
        self.exp_path = exp_path
        if save_path is None:
            self.save_path = self.exp_path
        else:
            self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.artifacts_file = self.save_path / "processed_artifacts.pkl"
        self.metrics_file = self.save_path / "artifact_metrics.pkl"
        self.expansion_file = self.save_path / "expansion_map.pkl"
        self.novelty_file = self.save_path / "novelties.pkl"

        self.replace_numbers = replace_numbers
        self.embed_names = embed_names

        self.world_log = None
        self.artifacts_by_ts = {}  # These are the active artifacts at each timestamp
        self.artifacts_by_creation = {}
        self.all_artifacts = {}
        self.embeddings = []
        self.metrics = {}
        self.expansion_map = None
        self.map_additions = None

        # Incremental update state — used by update_from_events().
        # _artifact_tag is the monotonic integer key for all_artifacts entries
        # and cannot be derived from existing state, so it must be persisted.
        self._artifact_tag: int = 0

        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.expansion_epsilon = expansion_epsilon
        self.expansion_metric = expansion_metric

        self.last_ts: int = -1

    def get_embeddings(self) -> np.ndarray:
        if not self.embeddings:
            self.embeddings = [art["embedding"] for art in self.all_artifacts.values()]
        return np.array(self.embeddings)

    def _replace_numbers(self, text: str) -> str:
        return re.sub(r"\d+", "X", text)

    def _embed_artifacts(self):
        artifacts = []
        for tag, artifact in self.all_artifacts.items():
            if self.embed_names:
                artifact_str = f"{artifact['name']}: {artifact['payload']}"
            else:
                artifact_str = str(artifact["payload"])
            if self.replace_numbers:
                artifact_str = self._replace_numbers(artifact_str)
            artifacts.append(artifact_str)
            self.all_artifacts[tag]["string"] = artifact_str

        dimensions = (
            NOT_GIVEN
            if self.embedding_dimensions is None
            else self.embedding_dimensions
        )

        start = 0
        interval = 1000
        prompt_tokens = 0
        total_tokens = 0

        for end in range(interval, len(artifacts) + interval, interval):
            end = min(len(artifacts), end)
            arts = artifacts[start:end]
            try:
                embs = openai.embeddings.create(
                    model=self.embedding_model,
                    input=arts,
                    dimensions=dimensions,
                )
            except Exception as e:
                log.warning("Error embedding artifacts %d to %d: %s", start, end, e)
                log.debug("%s", arts)
                raise e
            self.embeddings += [np.array(emb.embedding) for emb in embs.data]
            prompt_tokens += embs.usage.prompt_tokens
            total_tokens += embs.usage.total_tokens

            start = end
        log.info("Tokens used - Prompt tokens: %d - Total tokens: %d", prompt_tokens, total_tokens)

        for tag, emb in zip(self.all_artifacts, self.embeddings):
            self.all_artifacts[tag]["embedding"] = emb

        all_embedded = self._verify_embeddings()
        log.info("All artifacts embedded: %s", all_embedded)

    def _verify_embeddings(self):
        for tag in self.all_artifacts:
            if not (
                "embedding" in self.all_artifacts[tag]
                and self.all_artifacts[tag]["embedding"] is not None
            ):
                return False
        return True

    def find_artifact(
        self,
        name: str | None,
        payload: str | None,
        current_time: int | None,
        creator: str | None,
    ) -> dict:
        """Find an artifact by its name, payload, creation time, and creator.

        Args:
            name (str | None): The name of the artifact.
            payload (str | None): The payload of the artifact.
            current_time (str | None): The time step. If passed, it looks for all artifacts active at that timestep.
            creator (str | None): The creator of the artifact.
        """
        candidate_tags = []
        if current_time is not None:
            for t in range(0, current_time + 1):
                candidate_tags.extend(self.artifacts_by_ts[t])
            candidate_tags = list(set(candidate_tags))
        else:
            candidate_tags = list(self.all_artifacts.keys())

        for tag in candidate_tags:
            art = self.all_artifacts[tag]
            if name is not None and art["name"] != name:
                continue
            if payload is not None and art["payload"] != payload:
                continue
            if creator is not None and art["original_creator_tag"] != creator:
                continue
            return art
        return {}

    @staticmethod
    def _flatten_artifacts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flatten current + past versions into a single list.
        Each version includes a pointer to its immediately previous version.
        """
        flattened = []

        for item in items:
            name = item["name"]

            # Collect all versions: past + current
            versions = list(item.get("past_versions", []))
            versions.append(
                {
                    "name": name,
                    "payload": item["payload"],
                    "version": item["version"],
                }
            )

            # Sort by version number
            versions.sort(key=lambda v: v["version"])

            # Build flattened output
            for i, v in enumerate(versions):
                previous_version = None
                if i > 0:
                    pv = versions[i - 1]
                    previous_version = {
                        "name": pv["name"],
                        "payload": pv["payload"],
                        "version": pv["version"],
                    }

                flattened.append(
                    {
                        "name": v["name"],
                        "payload": v["payload"],
                        "version": v["version"],
                        "previous_version": previous_version,
                    }
                )

        return flattened

    def _load_raw_artifacts(self, last_ts: int):
        log.info("Loading raw artifacts")
        self.world_log = load_worldlog(self.exp_path / "open_gridworld.log", None, None)

        self.artifacts_by_ts = {}
        self.all_artifacts = {}
        self._artifact_tag = 0
        self.last_ts = -1

        self.update_from_events(self.world_log, last_ts)

    def load_full_history(self) -> int:
        """Read the full world log and rebuild the artifact index.

        Used on anthropologist restart to seed the index with every artifact
        ever created — the live log_scanner reads via byte offsets and only
        emits NEW events, so without this step artifacts created before the
        restart are invisible to name-based lookups (e.g. phylogeny requests
        from the dashboard).

        Returns the number of artifacts loaded.
        """
        world_log = load_worldlog(self.exp_path / "open_gridworld.log", None, None)
        last_ts = max(
            (int(e.get("timestamp", 0)) for e in world_log),
            default=0,
        )
        self.artifacts_by_ts = {}
        self.all_artifacts = {}
        self._artifact_tag = 0
        self.last_ts = -1
        self.update_from_events(world_log, last_ts)
        return len(self.all_artifacts)

    def update_from_events(self, raw_events: list[dict], up_to_ts: int) -> list[int]:
        """
        Incrementally update all_artifacts and artifacts_by_ts from pre-parsed
        world log events, without re-reading the log file from disk.

        Args:
            raw_events: List of raw world log event dicts (full obj, not filtered).
                        Only ARTIFACT_ADDED, ARTIFACT_INTERACTION(modify), and
                        ARTIFACT_REMOVED events are processed; others are ignored.
            up_to_ts: Fill artifacts_by_ts up to this timestep.

        Returns:
            List of new artifact tags added in this call (for phylogeny queuing).
        """
        active_tags = list(self.artifacts_by_ts.get(self.last_ts, []))
        new_tags: list[int] = []

        # Fill artifacts_by_ts for any timesteps between last_ts and the first
        # event's timestamp (gap between last update and now).
        ts = self.last_ts + 1

        for obj in sorted(raw_events, key=lambda e: int(e.get("timestamp", 0))):
            event = obj.get("event", "")
            event_ts = int(obj.get("timestamp", 0))

            if event_ts <= self.last_ts:
                # Already processed in a previous cycle — skip.
                # (Scan window is larger than detect_interval so events overlap.)
                continue

            # Fill in all timesteps up to this event
            while ts < event_ts:
                self.artifacts_by_ts[ts] = deepcopy(active_tags)
                ts += 1

            if event == "ARTIFACT_ADDED":
                art = obj.get("artifact", {})
                art["past_versions_tags"] = []
                art["version"] = 0
                art["original_version_creation"] = event_ts
                art["event"] = "created"
                art["original_creator_tag"] = art["creator_tag"]
                self.all_artifacts[self._artifact_tag] = deepcopy(art)
                active_tags.append(self._artifact_tag)
                new_tags.append(self._artifact_tag)
                self._artifact_tag += 1

            elif (
                event == "ARTIFACT_INTERACTION"
                and "modify_artifact" in obj.get("action", "")
                and "failed" not in obj.get("result", "").lower()
                and "unknown action" not in obj.get("result", "").lower()
            ):
                new_art = obj.get("artifact", {})
                latest_version = new_art["past_versions"][0]
                for ver in new_art["past_versions"]:
                    if ver["version"] > latest_version["version"]:
                        latest_version = ver
                old_name = latest_version["name"]
                old_payload = latest_version["payload"]

                old_art = None
                old_ii = None
                for ii, tag in enumerate(active_tags):
                    candidate = self.all_artifacts[tag]
                    if (
                        candidate["name"] == old_name
                        and candidate["payload"] == old_payload
                    ):
                        old_art = candidate
                        old_ii = ii
                        break
                if old_art is None:
                    action = obj.get("action", "")
                    if action == "modify_artifact":
                        old_name = str(obj.get("action_params", {}).get("name", ""))
                    else:
                        old_name = action.removeprefix("modify_artifact_")
                    for ii, tag in enumerate(active_tags):
                        candidate = self.all_artifacts[tag]
                        if candidate["name"] == old_name:
                            old_art = candidate
                            old_ii = ii
                            break

                if old_art is not None and old_ii is not None:
                    prev_tag = active_tags[old_ii]
                    new_art["previous_version_tag"] = prev_tag
                    new_art["version"] = old_art["version"] + 1
                    new_art["past_versions_tags"] = old_art["past_versions_tags"] + [
                        prev_tag
                    ]
                    new_art["creation_time"] = event_ts
                    new_art["event"] = "modified"
                    new_art["creator_tag"] = obj.get("agent_tag", "")
                    new_art["original_creator_tag"] = old_art.get("original_creator_tag", old_art.get("creator_tag", ""))
                    self.all_artifacts[self._artifact_tag] = deepcopy(new_art)
                    active_tags[old_ii] = self._artifact_tag
                    new_tags.append(self._artifact_tag)
                    self._artifact_tag += 1

            elif event == "ARTIFACT_REMOVED":
                ev_art = obj.get("artifact", {})
                for ii, tag in enumerate(active_tags):
                    act_art = self.all_artifacts[tag]
                    if (
                        act_art["name"] == ev_art["name"]
                        and act_art["payload"] == ev_art["payload"]
                    ):
                        active_tags.pop(ii)
                        break

        # Fill remaining timesteps up to up_to_ts
        while ts <= up_to_ts:
            self.artifacts_by_ts[ts] = deepcopy(active_tags)
            ts += 1

        self.last_ts = up_to_ts
        return new_tags

    def _load_processed_artifacts(self):
        data = pkl.load(open(self.artifacts_file, "rb"))
        self.artifacts_by_ts = data["artifacts_by_ts"]
        self.all_artifacts = data["all_artifacts"]

    def _save_processed_artifacts(self):
        if not self.all_artifacts or not self.artifacts_by_ts:
            log.info("No artifacts to save.")
            return
        data = {
            "artifacts_by_ts": self.artifacts_by_ts,
            "all_artifacts": self.all_artifacts,
        }
        with open(self.artifacts_file, "wb") as f:
            pkl.dump(data, f)
            log.info("Processed artifacts saved at: %s", self.artifacts_file)

    def _load_expansion_map(self):
        self.expansion_map = ExpansionMap(
            dimension=self.embedding_dimensions, epsilon=self.expansion_epsilon
        )
        data = pkl.load(open(self.expansion_file, "rb"))
        self.map_additions = data["map_additions"]
        self.expansion_map.from_dict(data["expansion_map"])

        # Now load the leaders embeddings
        for leader_id in self.expansion_map.leader_ids:
            emb = self.all_artifacts[leader_id]["embedding"]
            # Ensure the embedding is on the correct device
            emb_tensor = torch.Tensor(emb).to(self.expansion_map.device).reshape(1, -1)
            if self.expansion_map.leaders.numel() == 0:
                self.expansion_map.leaders = emb_tensor.clone()
            else:
                self.expansion_map.leaders = torch.cat(
                    [self.expansion_map.leaders, emb_tensor], dim=0
                )

    def _save_expansion_map(self):
        if self.expansion_map is None or self.map_additions is None:
            log.info("No expansion map to save.")
            return
        data = {
            "map_additions": self.map_additions,
            "expansion_map": self.expansion_map.as_dict(),
        }
        with open(self.expansion_file, "wb") as f:
            pkl.dump(data, f)
            log.info("Processed map additions saved at: %s", self.expansion_file)

    def _load_metrics(self):
        self.metrics = pkl.load(open(self.metrics_file, "rb"))

    def save_metrics(self):
        if not self.metrics:
            log.info("No metrics to save.")
            return
        with open(self.metrics_file, "wb") as f:
            pkl.dump(self.metrics, f)
            log.info("Metrics saved at: %s", self.metrics_file)

    # ----------- API ----------------

    def calc_expansion_map(self):
        log.info("Calculating expansion map...")
        self.expansion_map = ExpansionMap(
            dimension=self.embedding_dimensions,
            epsilon=self.expansion_epsilon,
            metric=self.expansion_metric,
        )
        self.map_additions = []
        for ts, tags in tqdm(self.get_artifact_by_creation().items(), desc="TS: "):
            added = 0
            for tag in tags:
                artifact = self.all_artifacts[tag]
                added += self.expansion_map.add_point(
                    new_point=artifact["embedding"], point_id=tag, timestamp=ts
                )
            self.map_additions.append(added)
        log.info("Expansion map done.")

    def artifact_lineage(self, tag: int) -> List[int]:
        """Given the artifact it follows it back until the origin
        Returns the list of artifact tags
        """
        chain = [tag]
        cur = self.all_artifacts[tag]
        while "previous_version_tag" in cur and cur["previous_version_tag"] is not None:
            prev = cur["previous_version_tag"]
            chain.append(prev)
            cur = self.all_artifacts[prev]
        chain.reverse()
        return chain

    def load(
        self,
        force_recalc: bool = False,
    ):
        """Loads artifacts, embeddings, expansion map, and metrics.
        If force_recalc is True, recomputes everything from raw data (except metrics).
        """
        self.last_ts = get_last_ts(self.exp_path / "open_gridworld.log")

        if self.artifacts_file.exists() and not force_recalc:
            log.info("Preprocessed artifacts found. Loading...")
            self._load_processed_artifacts()
            if not self._verify_embeddings():
                log.info("Some artifacts are missing embeddings. Re-embedding...")
                self._embed_artifacts()
                self._save_processed_artifacts()
        else:
            self._load_raw_artifacts(last_ts=self.last_ts)
            log.info("Processing artifacts...")
            self._embed_artifacts()
            self._save_processed_artifacts()

        self._artifact_tag = max(self.all_artifacts.keys(), default=-1) + 1

        if self.expansion_file.exists() and not force_recalc:
            log.info("Preprocessed expansion map found. Loading...")
            self._load_expansion_map()
        else:
            log.info("Creating expansion map...")
            self.calc_expansion_map()
            self._save_expansion_map()

        if self.metrics_file.exists() and not force_recalc:
            log.info("Preprocessed metrics found. Loading...")
            self._load_metrics()

        if self.novelty_file.exists() and not force_recalc:
            log.info("Preprocessed novelties found. Loading...")
            with open(self.novelty_file, "rb") as f:
                novelties = pkl.load(f)
            for tag in self.all_artifacts:
                self.all_artifacts[tag]["novelty"] = novelties[tag]

        log.info("Done.")

    def save(self):
        self._save_processed_artifacts()
        self._save_expansion_map()
        self.save_metrics()
        self.save_as_list()

    def save_as_list(self):
        arts_by_creation = self.get_artifact_by_creation()

        artifacts_list = []
        for ts, tags in arts_by_creation.items():
            for tag in tags:
                art_entry = {
                    "tag": tag,
                    "creation_time": ts,
                    "name": self.all_artifacts[tag]["name"],
                    "payload": self.all_artifacts[tag]["payload"],
                    "llm_novelty": self.all_artifacts[tag].get("novelty", None),
                }
                for metric in self.metrics:
                    art_entry[metric] = self.all_artifacts[tag].get(metric, None)
                artifacts_list.append(art_entry)

        list_df = pd.DataFrame(artifacts_list)
        save_path = self.save_path / "artifacts_list.csv"
        list_df.to_csv(save_path, index=False)
        log.info("Artifacts list saved at: %s", save_path)

    def _extract_artifacts_by_creation(self) -> dict:
        artifacts_by_creation = {}
        all_created = []
        for t, arts in self.artifacts_by_ts.items():
            new_arts = [a for a in arts if a not in all_created]
            all_created.extend(new_arts)
            artifacts_by_creation[t] = new_arts
        return artifacts_by_creation

    def get_artifact_by_creation(self, force: bool = False) -> dict:
        if not self.artifacts_by_creation or force:
            self.artifacts_by_creation = self._extract_artifacts_by_creation()
        return self.artifacts_by_creation


if __name__ == "__main__":
    arts = ExperimentArtifacts(LOGS_DIR / "PAPER_base_exp_1")
    arts.load(force_recalc=False)
