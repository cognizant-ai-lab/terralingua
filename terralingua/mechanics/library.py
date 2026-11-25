"""The shared library: artifacts every being can list, search, read, update,
deposit, retrieve and delete.

A preset attaches it through the run config:

    run:
      mechanics:
        terralingua.mechanics.library:
          max_size: -1
          notifications: all

The library keeps the artifacts it holds out of every cell, node and inventory.
It ages them, keeps their versions, and tells the beings what changed in it after
each step. Who used an artifact comes from the artifact's own record.
"""

import math
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from terralingua.environment.env_logger import Event
from terralingua.environment.mechanic import Mechanic

PASSAGE_CHARS = 120
CANDIDATES = 3

# ---- Texts shown to the beings. Giuseppe reviews them before an experiment. ----

INSTRUCTIONS = """=== The library ===
The library is the knowledge repository of the collective of agents.
Every agent can interact with it and add or read knowledge from it.
Knowledge is added to the library by depositing artifacts in it.
Using the library allows you and other agents to share knowledge,
keep track of changes, and access the collective intelligence of the group,
without duplicating effort or losing track of contributions.
Use it. Deposit what the others will need to know.
Read it before you decide, so you do not repeat work that is already done.
Avoid duplicating effort or ignoring the contributions of others."""
INSTRUCTIONS_CAP = "The library holds at most {max_size} artifacts."
INSTRUCTIONS_NOTIFY = (
    "Your observation lists the changes made to the library in the previous step."
)
INSTRUCTIONS_LIFESPAN = """Your life is limited. Nothing you learn survives your death except what is in the library.
The agent who needs your finding may not have been born yet."""

ACTIONS = {
    "show_library_content": {
        "description": "List the artifacts in the library. Each line gives the artifact's name, creator, creation step, last change, users, size and expiry.",
        "params": {
            "creator": "Name of a being. Lists only its artifacts.",
            "since": "A step number. Lists only artifacts created or changed at or after it.",
        },
        "optional": ["creator", "since"],
    },
    "search_library": {
        "description": "Find the library artifacts whose name or text contain the words of a query. Returns the best matches with their metadata and the matching passage.",
        "params": {
            "query": "The words to look for.",
            "limit": "Maximum number of results. Default {search_limit}.",
            "creator": "Name of a being. Searches only its artifacts.",
            "since": "A step number. Searches only artifacts created or changed at or after it.",
        },
        "optional": ["limit", "creator", "since"],
    },
    "read_artifact": {
        "description": "Read the full text of a library artifact. It stays in the library.",
        "params": {"name": "Name of an artifact in the library."},
    },
    "update_artifact": {
        "description": "Replace the text of a library artifact. The earlier text is kept as a version. When several agents update the same artifact in the same step, only one change is applied. The others are refused and receive the text of the accepted update.",
        "params": {
            "name": "Name of an artifact in the library.",
            "payload": "The new text.",
        },
    },
    "show_versions": {
        "description": "List the earlier versions of a library artifact: who changed it, when, and the text of each.",
        "params": {"name": "Name of an artifact in the library."},
    },
    "delete_artifact": {
        "description": "Remove an artifact from the library for good.",
        "params": {"name": "Name of an artifact in the library."},
    },
    "deposit_artifact": {
        "description": "Put one of your movable artifacts into the library.",
        "params": {"name": "Name of one of your movable artifacts."},
    },
    "retrieve_artifact": {
        "description": "Take an artifact out of the library onto your position.",
        "params": {"name": "Name of an artifact in the library."},
    },
}

LINE = "{name} | by {creator} | creation step {created} | last change step {changed} by {editor} | {accessed} | {tokens} tokens | {expiry}"
ACCESSED_NONE = "accessed by nobody else"
ACCESSED_SOME = "accessed by {count} other being(s), last step {last}"
EXPIRY_NONE = "no expiry"
EXPIRY_SOME = "expires in {steps} step(s)"
COLUMNS = "Columns: name | creator | creation step | last change by whom | accessed by others | size | expiry"
LISTING_HEADER = "Library: {count} artifact(s). " + COLUMNS
LISTING_EMPTY = "The library is empty."
LISTING_NO_MATCH = "No artifact matches the filters."
SEARCH_HEADER = 'Search "{query}": {count} result(s). ' + COLUMNS
SEARCH_NONE = 'No artifact matches "{query}".'
SEARCH_PASSAGE = "  …{passage}…"
READ = "{name} (v{version}):\n{text}"
VERSIONS_HEADER = "{name}: {count} earlier version(s)"
VERSIONS_NONE = "{name} has no earlier version."
VERSION = "v{version} | step {step} by {author} | replaced by {editor}\n{text}"
UPDATED = "Artifact {name} updated (v{version})."
NOT_EDITABLE = "Refused. Artifacts of type {type} cannot be updated."
DELETED = "Artifact {name} deleted."
DEPOSITED = "Artifact {name} deposited."
RETRIEVED = "Artifact {name} is now on your position."
NOT_IN_LIBRARY = "No artifact named {name} in the library."
NOT_DEPOSITABLE = "{name} is not one of your movable artifacts."
CONFLICT = (
    "Refused. {editor} already changed {name} in this step. Current text:\n{text}"
)
CONFLICT_GONE = (
    "Refused. {editor} already removed {name} from the library in this step."
)
INERT = "Artifacts are inert and cannot be read or moved."
EMPTY_QUERY = "Give one or more words to search for."
FULL = "Refused. The library is full ({count} artifacts). Oldest: {oldest}. Least read: {least_read}."

CREATE_ARTIFACT_NOTE = "Once it is deposited in the library, every being can search, read, retrieve, or update it."
NOTE_CHANGES = "Library changes"
NOTE_LIBRARY = "Library"
EVENT_LINES = {
    "deposited": "{actor} deposited {name}: {line}",
    "updated": "{actor} updated {name}: {line}",
    "deleted": "{actor} deleted {name}.",
    "retrieved": "{actor} retrieved {name}.",
    "expired": "{name} expired.",
    "kept": "{name} entered the library when {actor} died: {line}",
}

# ---- End of the texts. ----


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_size: int = Field(
        -1, ge=-1, description="Most artifacts the library holds. -1 means no limit."
    )
    notifications: Literal["none", "all", "followers"] = Field(
        "all",
        description="Who learns of a change in the library. none: nobody. all: every being. followers: the beings that receive the broadcasts of the being that made the change, its followers in the social graph and the beings within perception range elsewhere.",
    )
    search_limit: int = Field(
        5, ge=1, description="Results of search_library when the being gives no limit."
    )


def build(options: Options) -> List[Mechanic]:
    return [Library(options)]


def step_number(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def passage(text: str, words: List[str]) -> str:
    """The part of a text around the first word of the query that appears in it."""
    lowered = text.lower()
    hits = [lowered.find(word) for word in words if word and lowered.find(word) != -1]
    start = max(0, min(hits) - PASSAGE_CHARS) if hits else 0
    return " ".join(text[start : start + 2 * PASSAGE_CHARS].split())


class Library(Mechanic):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.written: Dict[str, str] = {}
        self.events: List[dict] = []

    # ---- the hooks ----

    def instructions(self, env) -> str:
        lines = [INSTRUCTIONS]
        if env.lifespan != -1:
            lines.append(INSTRUCTIONS_LIFESPAN)
        if self.options.max_size >= 0:
            lines.append(INSTRUCTIONS_CAP.format(max_size=self.options.max_size))
        if self.options.notifications != "none":
            lines.append(INSTRUCTIONS_NOTIFY)
        return "\n".join(lines)

    def on_menu(self, env, tag: str, menu: dict) -> dict:
        menu["show_library_content"] = self.entry("show_library_content")
        if self.stored(env) and not env.inert_artifacts:
            for action in ACTIONS:
                if action not in ("show_library_content", "deposit_artifact"):
                    menu[action] = self.entry(action)
        if self.depositable(env, tag) and not env.inert_artifacts:
            menu["deposit_artifact"] = self.entry("deposit_artifact")
        if "create_artifact" in menu:
            entry = menu["create_artifact"]
            menu["create_artifact"] = {
                **entry,
                "description": f"{entry['description']} {CREATE_ARTIFACT_NOTE}",
            }
        return menu

    def on_action(self, env, tag: str, action: str, params: dict) -> str | None:
        if action not in ACTIONS:
            return None
        if env.inert_artifacts and action != "show_library_content":
            return INERT
        return getattr(self, action)(env, tag, params or {})

    def on_step(self, env, infos: dict) -> None:
        for name in self.stored(env):
            if not env.age_artifact(name):
                env.expire_artifact(name)
                self.record(env, "expired", name)
        self.notify(env)
        self.written = {}

    def on_death(self, env, tag: str, artifacts: list) -> None:
        for name in sorted(artifacts):
            if name not in env.artifacts or self.full(env):
                continue
            self.store(env, name, tag)
            self.record(env, "kept", name, tag)
            self.log(
                env, Event.LIBRARY_DEPOSIT, tag, name, "Success", reason="owner_died"
            )
        self.notify(env)

    def on_birth(self, env, tag: str) -> None:
        names = self.stored(env)
        if names:
            env.note(tag, NOTE_LIBRARY, self.listing(env, self.ordered(env, names)))

    # ---- the actions ----

    def show_library_content(self, env, tag: str, params: dict) -> str:
        names = self.ordered(env, self.filtered(env, params))
        if not self.stored(env):
            return LISTING_EMPTY
        if not names:
            return LISTING_NO_MATCH
        return self.listing(env, names)

    def search_library(self, env, tag: str, params: dict) -> str:
        query = str(params.get("query") or "").strip()
        if not query:
            return EMPTY_QUERY
        limit = step_number(params.get("limit")) or self.options.search_limit
        names = self.filtered(env, params)
        ranked = self.rank(env, names, query)[: max(1, limit)]
        self.log(
            env,
            Event.LIBRARY_SEARCH,
            tag,
            None,
            "Success",
            query=query,
            results=len(ranked),
        )
        if not ranked:
            return SEARCH_NONE.format(query=query)
        words = query.lower().split()
        lines = [SEARCH_HEADER.format(query=query, count=len(ranked))]
        for name in ranked:
            lines.append(self.line(env, name))
            lines.append(
                SEARCH_PASSAGE.format(
                    passage=passage(str(env.artifacts[name].payload), words)
                )
            )
        return "\n".join(lines)

    def read_artifact(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if name not in self.stored(env):
            return NOT_IN_LIBRARY.format(name=name)
        artifact = env.artifacts[name]
        artifact.users[tag].add(env.step_count)
        self.log(env, Event.LIBRARY_READ, tag, name, "Success")
        return READ.format(name=name, version=artifact.version, text=artifact.payload)

    def update_artifact(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if refusal := self.refusal(env, tag, name):
            return refusal
        artifact = env.artifacts[name]
        if "modify_artifact" not in artifact.actions:
            return NOT_EDITABLE.format(type=artifact.art_type)
        before = artifact.version
        message = artifact.interact(
            tag,
            "modify_artifact",
            {"payload": params.get("payload", "")},
            env.step_count,
        )
        if artifact.version == before:
            self.log(env, Event.LIBRARY_UPDATE, tag, name, message)
            return message
        self.written[name] = tag
        self.record(env, "updated", name, tag)
        self.log(env, Event.LIBRARY_UPDATE, tag, name, "Success")
        return UPDATED.format(name=name, version=artifact.version)

    def show_versions(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if name not in self.stored(env):
            return NOT_IN_LIBRARY.format(name=name)
        past = env.artifacts[name].past_versions
        self.log(env, Event.LIBRARY_VERSIONS, tag, name, "Success")
        if not past:
            return VERSIONS_NONE.format(name=name)
        lines = [VERSIONS_HEADER.format(name=name, count=len(past))]
        author = env.artifacts[name].creator
        for entry in past:
            lines.append(
                VERSION.format(
                    version=entry.get("version"),
                    step=entry.get("version_creation_time"),
                    author=self.being(env, author),
                    editor=self.being(env, entry.get("modified_by")),
                    text=entry.get("payload"),
                )
            )
            author = entry.get("modified_by")
        return "\n".join(lines)

    def delete_artifact(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if refusal := self.refusal(env, tag, name):
            return refusal
        env.expire_artifact(name, reason="deleted", agent_tag=tag)
        self.written[name] = tag
        self.record(env, "deleted", name, tag)
        self.log(env, Event.LIBRARY_DELETE, tag, name, "Success")
        return DELETED.format(name=name)

    def deposit_artifact(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if name not in self.depositable(env, tag):
            return NOT_DEPOSITABLE.format(name=name)
        if self.full(env):
            return self.full_refusal(env)
        self.store(env, name, tag)
        self.record(env, "deposited", name, tag)
        self.log(env, Event.LIBRARY_DEPOSIT, tag, name, "Success")
        return DEPOSITED.format(name=name)

    def retrieve_artifact(self, env, tag: str, params: dict) -> str:
        name = params.get("name")
        if refusal := self.refusal(env, tag, name):
            return refusal
        env.place_artifact(name, env.agent_pos[tag])
        env.artifacts[name].users[tag].add(env.step_count)
        self.written[name] = tag
        self.record(env, "retrieved", name, tag)
        self.log(
            env, Event.LIBRARY_RETRIEVE, tag, name, "Success", pose=env.agent_pos[tag]
        )
        return RETRIEVED.format(name=name)

    # ---- what the library holds ----

    def stored(self, env) -> List[str]:
        return [
            name
            for name, (kind, holder) in env.artifact_location.items()
            if kind == "stored" and holder == self.name
        ]

    def store(self, env, name: str, tag: str) -> None:
        env.detach_artifact(name, self.name)
        env.artifacts[name].users[tag].add(env.step_count)

    def full(self, env) -> bool:
        return 0 <= self.options.max_size <= len(self.stored(env))

    def depositable(self, env, tag: str) -> List[str]:
        held = set(env.agent_inventories.get(tag, set())) | set(
            env.pos_artifacts.get(env.agent_pos.get(tag), set())
        )
        return sorted(
            name
            for name in held
            if name in env.artifacts and env.artifacts[name].movable
        )

    def filtered(self, env, params: dict) -> List[str]:
        creator = str(params.get("creator") or "").strip()
        wanted = env.name_to_tag.get(creator, creator)
        since = step_number(params.get("since"))
        names = []
        for name in self.stored(env):
            artifact = env.artifacts[name]
            if creator and artifact.creator != wanted:
                continue
            if since is not None and artifact.version_creation_time < since:
                continue
            names.append(name)
        return names

    def ordered(self, env, names: List[str]) -> List[str]:
        return sorted(
            names, key=lambda name: (-env.artifacts[name].version_creation_time, name)
        )

    def rank(self, env, names: List[str], query: str) -> List[str]:
        if not names:
            return []
        documents = [f"{name}\n{env.artifacts[name].payload}" for name in names]
        vectorizer = TfidfVectorizer()
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError:
            return []
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        scored = sorted(zip(scores, names), key=lambda item: (-item[0], item[1]))
        return [name for score, name in scored if score > 0]

    # ---- texts ----

    def entry(self, action: str) -> dict:
        spec = ACTIONS[action]
        entry = {
            "description": spec["description"],
            "params": {
                key: text.format(search_limit=self.options.search_limit)
                for key, text in spec["params"].items()
            },
        }
        if spec.get("optional"):
            entry["optional"] = list(spec["optional"])
        return entry

    def being(self, env, tag) -> str:
        return env.agent_names.get(tag, str(tag))

    def line(self, env, name: str) -> str:
        artifact = env.artifacts[name]
        editor = (
            artifact.past_versions[-1]["modified_by"]
            if artifact.past_versions
            else artifact.creator
        )
        uses = self.uses(artifact)
        accessed = (
            ACCESSED_SOME.format(
                count=len(uses), last=max(max(steps) for steps in uses.values())
            )
            if uses
            else ACCESSED_NONE
        )
        remaining = artifact.remaining_time
        expiry = (
            EXPIRY_NONE
            if remaining is None or math.isinf(remaining)
            else EXPIRY_SOME.format(steps=int(remaining))
        )
        return LINE.format(
            name=name,
            creator=self.being(env, artifact.creator),
            created=artifact.creation_time,
            changed=artifact.version_creation_time,
            editor=self.being(env, editor),
            accessed=accessed,
            tokens=artifact.token_count(),
            expiry=expiry,
        )

    def uses(self, artifact) -> Dict[str, set]:
        """The artifact's own record of who used it, without its creator."""
        return {
            tag: steps
            for tag, steps in artifact.users.items()
            if tag != artifact.creator and steps
        }

    def listing(self, env, names: List[str]) -> str:
        return "\n".join(
            [
                LISTING_HEADER.format(count=len(names)),
                *(self.line(env, name) for name in names),
            ]
        )

    def refusal(self, env, tag: str, name) -> str | None:
        """Why `tag` may not change `name` now: another being changed it in this
        step, or it is not in the library. None when it may."""
        if not isinstance(name, str):
            return NOT_IN_LIBRARY.format(name=name)
        editor = self.written.get(name)
        if editor is not None and editor != tag:
            if name in self.stored(env):
                return CONFLICT.format(
                    editor=self.being(env, editor),
                    name=name,
                    text=env.artifacts[name].payload,
                )
            return CONFLICT_GONE.format(editor=self.being(env, editor), name=name)
        if name not in self.stored(env):
            return NOT_IN_LIBRARY.format(name=name)
        return None

    def full_refusal(self, env) -> str:
        names = self.stored(env)
        oldest = sorted(
            names, key=lambda name: (env.artifacts[name].creation_time, name)
        )[:CANDIDATES]
        least_read = sorted(
            names, key=lambda name: (len(self.uses(env.artifacts[name])), name)
        )[:CANDIDATES]
        return FULL.format(
            count=len(names), oldest=", ".join(oldest), least_read=", ".join(least_read)
        )

    # ---- events, notes and the log ----

    def record(self, env, kind: str, name: str, actor: str | None = None) -> None:
        if self.options.notifications == "none":
            return
        line = self.line(env, name) if name in env.artifacts else ""
        self.events.append({"kind": kind, "name": name, "actor": actor, "line": line})

    def recipients(self, env, actor: str | None) -> List[str]:
        if self.options.notifications == "all" or actor is None:
            return sorted(tag for tag in env.agent_registry if tag != actor)
        return env.audience_of(actor)

    def notify(self, env) -> None:
        events, self.events = self.events, []
        for event in events:
            text = EVENT_LINES[event["kind"]].format(
                actor=self.being(env, event["actor"]),
                name=event["name"],
                line=event["line"],
            )
            for tag in self.recipients(env, event["actor"]):
                env.note(tag, NOTE_CHANGES, text)

    def log(
        self, env, event: Event, tag: str, name: str | None, status: str, **fields
    ) -> None:
        if env.logger:
            env.logger.log(
                time=env.step_count,
                event_type=event,
                agent_tag=tag,
                agent_name=env.agent_names.get(tag),
                artifact_name=name,
                status=status,
                **fields,
            )
