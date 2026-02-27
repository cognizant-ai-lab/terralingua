import json
import pickle as pkl
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import openai
import pandas as pd
import spacy
import torch
import zstandard as zstd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import NOT_GIVEN
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import ROOT
from core.utils.analysis_utils import get_last_ts, load_agent_log, load_worldlog

load_dotenv()


# Metrics
# ================================
@dataclass
class Metric:
    name: str

    def _evaluate(self, text: str):
        raise NotImplementedError

    def compute(self, artifacts: "ExperimentArtifacts"):
        print(f"Calculating {self.name}...")
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
                print(f"Error at ts {ts} with tags {tags}: {e}")
                metric_by_ts["mean"].append(0.0)
                metric_by_ts["std"].append(0.0)
                metric_by_ts["max"].append(0.0)
                metric_by_ts["min"].append(0.0)
                metric_by_ts["median"].append(0.0)
        artifacts.metrics[self.name] = metric_by_ts
        artifacts.save_metrics()
        print(f"{self.name} calculation done.")


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
        print("Loading Tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        print("Loading LM...")
        self.lm = (
            AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device).eval()  # type: ignore
        )
        print("Done")

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
        self.save_path = ROOT / "vectorizer.joblib"
        self._TOKEN_RE = re.compile(r"(?u)\b\w+\b")

        if self.save_path.exists():
            print("Vectorized dataset found. Loading...")
            self.vect = joblib.load(self.save_path)
        else:
            print("Creating vectorized dataset...")
            # Fit the vectorizer on the given corpus
            self.vect = TfidfVectorizer(
                lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=5
            )
            print("Loading data corpus...")
            ds = load_dataset(
                self.dataset_name, self.dataset_config, split=self.dataset_split
            )
            corpus = [re.sub(r"\s+", " ", x["text"]) for x in ds if x["text"].strip()]  # type: ignore
            print("Fitting vectorizer...")
            self.vect.fit(corpus)
            print("Saving...")
            joblib.dump(self.vect, self.save_path)
            print("Done.")

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
        self.novelty_file = self.save_path / "novelties.pkl"

        self.replace_numbers = replace_numbers
        self.embed_names = embed_names

        self.world_log = None
        self.artifacts_by_ts = {}  # These are the active artifacts at each timestamp
        self.artifacts_by_creation = {}
        self.all_artifacts = {}
        self.embeddings = []
        self.metrics = {}

        self.embedding_model = embedding_model
        self.embedding_dimensions = (
            embedding_dimensions if embedding_dimensions is not None else 512
        )

        self.last_ts = get_last_ts(worldlog_path=self.exp_path / "open_gridworld.log")

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
                print(f"Error embedding artifacts {start} to {end}: {e}")
                print(arts)
                raise e
            self.embeddings += [np.array(emb.embedding) for emb in embs.data]
            prompt_tokens += embs.usage.prompt_tokens
            total_tokens += embs.usage.total_tokens

            start = end
        print(
            f"Tokens used - Prompt tokens: {prompt_tokens} - Total tokens: {total_tokens}"
        )

        for tag, emb in zip(self.all_artifacts, self.embeddings):
            self.all_artifacts[tag]["embedding"] = emb

        all_embedded = self._verify_embeddings()
        print(f"All artifacts embedded: {all_embedded}")

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

    def _load_raw_artifacts(self):
        print("Loading raw artifacts")
        self.world_log = load_worldlog(self.exp_path / "open_gridworld.log", None, None)

        self.artifacts_by_ts = {}
        active_artifacts_tag = []
        self.all_artifacts = {}
        ts = 0
        artifact_tag = 0
        for event in self.world_log:
            # There can be multiple event lines with the same time step. We have to be sure to include all of them
            while ts < event["timestamp"]:
                self.artifacts_by_ts[ts] = deepcopy(active_artifacts_tag)
                ts += 1

            if ts > self.last_ts:
                break

            if event["event"] == "ARTIFACT_ADDED":
                art = event["artifact"]
                art["past_versions_tags"] = []
                art["version"] = 0
                art["original_version_creation"] = event["timestamp"]
                art["event"] = "created"
                art["original_creator_tag"] = art["creator_tag"]
                self.all_artifacts[artifact_tag] = deepcopy(art)
                active_artifacts_tag.append(artifact_tag)
                artifact_tag += 1

            # Modified artifacts are considered as created at that point, but keep the score of previous versions
            if (
                event["event"] == "ARTIFACT_INTERACTION"
                and "modify_artifact" in event["action"]
                and "failed" not in event["result"].lower()
                and "unknown action" not in event["result"].lower()
            ):
                # In case the name was modifiable
                new_art = event["artifact"]
                latest_version = event["artifact"]["past_versions"][0]
                for ver in event["artifact"]["past_versions"]:
                    if ver["version"] > latest_version["version"]:
                        latest_version = ver

                old_name = latest_version["name"]
                old_payload = latest_version["payload"]

                # Now find the old artifact among the active ones
                old_art = None
                for ii, tag in enumerate(active_artifacts_tag):
                    candidate_art = self.all_artifacts[tag]
                    if (
                        candidate_art["name"] == old_name
                        and candidate_art["payload"] == old_payload
                    ):
                        old_art = candidate_art
                        break
                # ---------------------------

                # If it cannot be found like that try to find it just by name
                # Reason: we tracked artifacts by name, but the unique name was not enforced cause of a bug,
                # This means that the new artifacts with same name would overwrite previous ones.
                # But if it was not found by the previous method, it means that we can find it by name through the active artifacts
                if old_art is None:
                    old_name = event["action"].removeprefix("modify_artifact_")
                    for ii, tag in enumerate(active_artifacts_tag):
                        candidate_art = self.all_artifacts[tag]
                        if candidate_art["name"] == old_name:
                            old_art = candidate_art
                            break
                # ---------------------------
                assert old_art is not None, (
                    f"Old artifact not found for modification. Old name: {old_name} - New name: {new_art['name']} \nEvent: {event}"
                )

                new_art["previous_version_tag"] = tag
                new_art["version"] = old_art["version"] + 1
                new_art["past_versions_tags"] = old_art["past_versions_tags"] + [tag]
                new_art["creation_time"] = event["timestamp"]
                new_art["event"] = "modified"
                new_art["creator_tag"] = event["agent_tag"]

                self.all_artifacts[artifact_tag] = deepcopy(new_art)
                # Replace old tag with new one
                active_artifacts_tag[ii] = artifact_tag
                artifact_tag += 1

            if event["event"] == "ARTIFACT_REMOVED":
                ev_art = event["artifact"]
                for ii, tag in enumerate(active_artifacts_tag):
                    act_art = self.all_artifacts[tag]
                    if (
                        act_art["name"] == ev_art["name"]
                        and act_art["payload"] == ev_art["payload"]
                    ):
                        active_artifacts_tag.pop(ii)
                        break

        # Add the last artifacts (need the <= !)
        while ts <= event["timestamp"]:
            self.artifacts_by_ts[ts] = deepcopy(active_artifacts_tag)
            ts += 1

    def _load_processed_artifacts(self):
        data = pkl.load(open(self.artifacts_file, "rb"))
        self.artifacts_by_ts = data["artifacts_by_ts"]
        self.all_artifacts = data["all_artifacts"]

    def _save_processed_artifacts(self):
        if not self.all_artifacts or not self.artifacts_by_ts:
            print("No artifacts to save.")
            return
        data = {
            "artifacts_by_ts": self.artifacts_by_ts,
            "all_artifacts": self.all_artifacts,
        }
        with open(self.artifacts_file, "wb") as f:
            pkl.dump(data, f)
            print(f"Processed artifacts saved at: {self.artifacts_file}")

    def _load_metrics(self):
        self.metrics = pkl.load(open(self.metrics_file, "rb"))

    def save_metrics(self):
        if not self.metrics:
            print("No metrics to save.")
            return
        save_path = self.save_path / "artifact_metrics.pkl"
        with open(save_path, "wb") as f:
            pkl.dump(self.metrics, f)
            print(f"Metrics saved at: {save_path}")

    # ----------- API ----------------

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
        If force recalc is True, it will recompute everything from raw data, except the metrics.

        Args:
            exp_path (Path | str): _description_
            model (str, optional): _description_. Defaults to "text-embedding-3-small".
            dimensions (_type_, optional): _description_. Defaults to None.
            force_recalc (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[Dict[int, List[int]], Dict[int, dict]]: artifacts_by_ts, all_artifacts
        """
        self._load_raw_artifacts()

        if self.artifacts_file.exists() and not force_recalc:
            print("Preprocessed artifacts found. Loading...")
            self._load_processed_artifacts()
            if not self._verify_embeddings():
                print("Some artifacts are missing embeddings. Re-embedding...")
                self._embed_artifacts()
                self._save_processed_artifacts()
        else:
            print("Processing artifacts...")
            self._embed_artifacts()
            self._save_processed_artifacts()

        if self.metrics_file.exists() and not force_recalc:
            print("Preprocessed metrics found. Loading...")
            self._load_metrics()

        if self.novelty_file.exists() and not force_recalc:
            print("Preprocessed novelties found. Loading...")
            with open(self.novelty_file, "rb") as f:
                novelties = pkl.load(f)
            for tag in self.all_artifacts:
                self.all_artifacts[tag]["novelty"] = novelties[tag]

        print("Done.")

    def save(self):
        self._save_processed_artifacts()
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
        print(f"Artifacts list saved at: {save_path}")

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
    arts = ExperimentArtifacts(ROOT / "logs" / "PAPER_base_exp_1")
    arts.load(force_recalc=False)
