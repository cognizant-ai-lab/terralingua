import inspect
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import tiktoken

MAX_TEXT_ARTIFACT_SIZE = 500

ARTIFACT_TYPES: Dict[str, type] = {}


def register_artifact_type(name: str):
    """Class decorator. Registers an Artifact subclass under a type name.

    The world looks the class up here to create, seed, and restore artifacts.
    Set `creatable = True` on the class to let agents create it.
    """

    def decorator(cls):
        cls.art_type = name
        ARTIFACT_TYPES[name] = cls
        return cls

    return decorator


def creatable_types() -> Dict[str, str]:
    """Type names agents may create, with their descriptions."""
    return {name: cls.description for name, cls in ARTIFACT_TYPES.items() if cls.creatable}


def describe_types() -> List[dict]:
    """Every registered type: name, description, whether beings may create it, and its own parameters.

    The parameters are the constructor arguments a type adds to the common ones.
    A seed file passes them under "params".
    """
    common = set(inspect.signature(Artifact.__init__).parameters)
    out = []
    for name, cls in sorted(ARTIFACT_TYPES.items()):
        params = [
            p for p in inspect.signature(cls.__init__).parameters
            if p not in common and p not in ("args", "kwargs")
        ]
        out.append({
            "name": name,
            "description": cls.description,
            "creatable": bool(cls.creatable),
            "params": params,
        })
    return out


ArtifactCreationError = ValueError


class Artifact:
    art_type: str = ""
    creatable: bool = False
    description: str = ""

    def __init__(
        self,
        name: str,
        payload: Any,
        lifespan: int | float,
        pose: Tuple[int, int] | str,
        creator: str,
        creation_time: int,
        movable: bool = True,
    ):
        self.name = name
        valid, error_message = self.verify_payload(payload=payload)
        if valid:
            self.payload = payload
        else:
            raise ArtifactCreationError(
                f"Invalid payload for artifact type {self.art_type}: {error_message}"
            )
        self.pose = pose
        self.creator = creator
        self.lifespan = lifespan
        self.remaining_time = np.inf if lifespan == -1 else lifespan
        # Agents that interfaced with it. Just for tracking
        self.users: Dict[str, Set[int]] = defaultdict(set)
        self.creation_time = creation_time
        self.version_creation_time = creation_time
        self.deletion_time: int | None = None
        self.past_versions = []
        self.version = 0
        self.movable = movable

    @property
    @abstractmethod
    def actions(self) -> dict:
        raise NotImplementedError("Must specify artifact actions")

    def serialize(self) -> dict:
        serialized = {
            "name": self.name,
            "art_type": self.art_type,
            "payload": self.payload,
            "lifespan": "inf" if self.lifespan == np.inf else self.lifespan,
            "pose": str(self.pose),
            "creator_tag": self.creator,
            "users_tag": {user: list(ts) for user, ts in self.users.items()},
            "creation_time": self.creation_time,
            "past_versions": self.past_versions,
            "version": self.version,
            "version_creation_time": self.version_creation_time,
            "movable": self.movable,
        }
        max_tokens = getattr(self, "max_tokens", None)
        if max_tokens is not None:
            serialized["max_tokens"] = max_tokens
        if self.deletion_time is not None:
            serialized["deletion_time"] = self.deletion_time
        else:
            serialized["remaining_time"] = (
                "inf" if self.remaining_time == np.inf else self.remaining_time
            )
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        name = data["name"]
        payload = data["payload"]
        lifespan = np.inf if data["lifespan"] == "inf" else int(data["lifespan"])
        pose = data["pose"]
        creator = data["creator_tag"]
        users = defaultdict(set)
        for user, ts in data["users_tag"].items():
            users[user] = set(ts)
        creation_time = data["creation_time"]
        if "deletion_time" in data:
            deletion_time = data["deletion_time"]
        else:
            remaining_time = (
                np.inf if data["remaining_time"] == "inf" else data["remaining_time"]
            )
        past_versions = data.get("past_versions", [])
        version = data.get("version", 0)
        movable = data.get("movable", True)

        # Only TextArtifact carries a configurable token cap; forward it so a
        # payload that was valid under a raised cap survives checkpoint restore.
        max_tokens = data.get("max_tokens")
        extra = {"max_tokens": max_tokens} if max_tokens is not None else {}
        artifact = cls(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
            movable=movable,
            **extra,
        )
        artifact.users = users
        artifact.deletion_time = deletion_time if "deletion_time" in data else None
        artifact.remaining_time = remaining_time if "remaining_time" in data else None
        artifact.past_versions = past_versions
        artifact.version = version
        return artifact

    @abstractmethod
    def interact(self, agent_name: str, action: str, params: dict, timestamp: int):
        raise NotImplementedError(
            "interact method not implemented for base Artifact class"
        )

    @abstractmethod
    def passive_effect(self, timestamp: int, agent_name: str):
        """This is the effect that the artifact has on the agents that just step on it"""
        raise NotImplementedError(
            "passive_effect method not implemented for base Artifact class"
        )

    @abstractmethod
    def verify_payload(self, payload) -> Tuple[bool, str]:
        """Verify that the payload is valid for the artifact type"""
        raise NotImplementedError(
            "verify_payload method not implemented for base Artifact class"
        )


@register_artifact_type("text")
class TextArtifact(Artifact):
    """An artifact that contains text.
    Agents can act on it by modifying its content or destroing the artifact.
    Passive effect: read the content
    """

    creatable = True
    description = f"Any alfanumeric data stored in a physical marker. Maximum size is {MAX_TEXT_ARTIFACT_SIZE} tokens."

    def __init__(
        self,
        name: str,
        payload: str,
        lifespan: int | float,
        pose: Tuple[int, int] | str,
        creator: str,
        creation_time: int,
        movable: bool = True,
        max_tokens: int = MAX_TEXT_ARTIFACT_SIZE,
    ):
        self.payload_encoder = tiktoken.get_encoding("cl100k_base")
        # Set before super().__init__, which calls verify_payload during construction.
        self.max_tokens = int(max_tokens)
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
            movable=movable,
        )
        self.creation_cost = 0

    @property
    def actions(self):
        return {
            "destroy_artifact": {
                "description": "Destroys an artifact",
                "params": {},
            },
            "modify_artifact": {
                "description": "Modifies the content of an artifact",
                "params": {
                    "payload": "New content of the artifact",
                    "lifespan": "New lifespan of the artifact",
                },
            },
        }

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        artifact_representation = f"""Artifact {self.name}\n\tCreated by {self.creator} at time {self.creation_time}"""

        if self.past_versions:
            artifact_representation += f"\n\tLast modified at time {self.version_creation_time} by {self.past_versions[-1]['modified_by']} (version {self.version})"

        artifact_representation += f"\n\tContent: {self.payload}"
        return artifact_representation

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        self.users[agent_name].add(timestamp)
        if action not in self.actions:
            return f"Unknown action: {action} - Available actions: {self.actions}"
        if action == "modify_artifact":
            if "movable" in params:
                return f"Failed to modify artifact {self.name}: 'movable' cannot be changed after creation"
            payload = params.get("payload", "")
            valid, error_message = self.verify_payload(payload)
            if not valid:
                return f"Failed to modify artifact {self.name}: {error_message}"

            # Validate the entire proposal before saving a version or replacing
            # content. A refused lifespan must not leave a partially applied edit.
            lifespan = self.lifespan
            if "lifespan" in params:
                try:
                    requested_lifespan = int(params["lifespan"])
                except (ValueError, TypeError, OverflowError):
                    return f"Failed to modify artifact {self.name}: 'lifespan' must be an integer"
                lifespan = np.inf if requested_lifespan == -1 else requested_lifespan

            # Do not change the name. This ensure uniqueness of the artifacts
            past_version = {
                "payload": self.payload,
                "lifespan": "inf" if self.lifespan == np.inf else self.lifespan,
                "name": self.name,
                "version": self.version,
                "version_creation_time": self.version_creation_time,
                "modified_by": agent_name,
            }
            self.past_versions.append(past_version)

            self.version += 1
            self.version_creation_time = timestamp
            self.payload = payload
            self.lifespan = lifespan
            self.remaining_time = np.inf if self.lifespan == -1 else self.lifespan
            return f"Artifact {self.name} updated"
        if action == "destroy_artifact":
            self.remaining_time = 0
            return f"Artifact {self.name} destroyed"
        return ""

    def verify_payload(self, payload) -> Tuple[bool, str]:
        payload = str(payload)
        if not isinstance(payload, str):
            return False, "Payload must be a string for TextArtifact"
        token_count = len(self.payload_encoder.encode(payload))
        max_tokens = getattr(self, "max_tokens", MAX_TEXT_ARTIFACT_SIZE)
        if token_count > max_tokens:
            return (
                False,
                f"Payload exceeds maximum token limit of {max_tokens} tokens (got {token_count} tokens)",
            )
        return True, ""


class ContainerArtifact(Artifact):
    """An artifact that can contain other artifacts.
    Agents can act on it by adding/removing artifacts, or destroing the container.
    Passive effect: list contents
    """

    def __init__(
        self,
        name: str,
        payload: List[str],
        lifespan: int | float,
        pose: Tuple[int, int] | str,
        creator: str,
        creation_time: int,
    ):
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "container"
        self.creation_cost = 5

    @property
    def actions(self):
        # TODO FOR CREATION ADD MAX CAPACITY
        # TODO MAKE THEM NON NESTABLE
        return {
            "destroy_artifact": {
                "description": "Destroys an artifact",
                "params": {},
            },
            "add_to_container": {
                "description": "Adds an artifact to a container",
                "params": {
                    "artifact": "Artifact to add to the container",
                },
            },
            "remove_from_container": {
                "description": "Removes an artifact from a container",
                "params": {
                    "artifact_name": "Name of the artifact to remove from the container",
                },
            },
        }


class ResourceModifierArtifact(Artifact):
    """An artifact that modifies food generation rates.
    Agents can act on it by changing its modifier or destroing the artifact.
    Passive effect: modify food generation rates when stepped on.
    """

    def __init__(
        self,
        name: str,
        payload: float,
        pose: Tuple[int, int] | str,
        creator: str,
        creation_time: int,
    ):
        super().__init__(
            name=name,
            payload=payload,
            lifespan=5,  # Food modifier artifacts have a fixed short lifespan
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "food_modifier"
        self.creation_cost = 10

    @property
    def actions(self):
        return {
            "destroy_artifact": {
                "description": "Destroys an artifact",
                "params": {},
            },
            "modify_artifact": {
                "description": "Modifies the food generation rate of an artifact",
                "params": {
                    "payload": "New food generation rate modifier",
                    "lifespan": "New lifespan of the artifact",
                },
            },
        }


class MovementModifierArtifact(Artifact):
    """An artifact that modifies agent movement speed.
    Agents can act on it by changing its modifier or destroing the artifact.
    Passive effect: modify agent movement speed when stepped on.
    """

    def __init__(
        self,
        name: str,
        payload: float,
        pose: Tuple[int, int] | str,
        creator: str,
        creation_time: int,
    ):
        super().__init__(
            name=name,
            payload=payload,
            lifespan=5,  # Movement modifier artifacts have a fixed short lifespan
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "movement_modifier"
        self.creation_cost = 10

    @property
    def actions(self):
        return {
            "destroy_artifact": {
                "description": "Destroys an artifact",
                "params": {},
            },
            "modify_artifact": {
                "description": "Modifies the movement speed of an artifact",
                "params": {
                    "payload": "New movement speed modifier",
                    "lifespan": "New lifespan of the artifact",
                },
            },
        }
