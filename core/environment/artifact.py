from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import tiktoken

MAX_TEXT_ARTIFACT_SIZE = 500

ARTIFACT_TYPE = {
    "text": f"Any alfanumeric data stored in a physical marker. Maximum size is {MAX_TEXT_ARTIFACT_SIZE} tokens.",
}

ArtifactCreationError = ValueError


class Artifact:
    def __init__(
        self,
        name: str,
        payload: Any,
        lifespan: int | float,
        pose: Tuple[int, int],
        creator: str,
        creation_time: int,
    ):
        self.name = name
        self.art_type = None
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
            "pose": (int(self.pose[0]), int(self.pose[1])),
            "creator_tag": self.creator,
            "users_tag": {user: list(ts) for user, ts in self.users.items()},
            "creation_time": self.creation_time,
            "past_versions": self.past_versions,
            "version": self.version,
            "version_creation_time": self.version_creation_time,
        }
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
        pose = (data["pose"][0], data["pose"][1])
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

        artifact = cls(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
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


class TextArtifact(Artifact):
    """An artifact that contains text.
    Agents can act on it by modifying its content or destroing the artifact.
    Passive effect: read the content
    """

    def __init__(
        self,
        name: str,
        payload: str,
        lifespan: int | float,
        pose: Tuple[int, int],
        creator: str,
        creation_time: int,
    ):
        self.payload_encoder = tiktoken.get_encoding("cl100k_base")
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "text"
        self.creation_cost = 0

    @property
    def actions(self):
        return {
            f"destroy_artifact_{self.name}": {
                "description": f"Destroys {self.name} artifact",
                "params": {},
            },
            f"modify_artifact_{self.name}": {
                "description": f"Modifies the content of {self.name} artifact",
                "params": {
                    "payload": "New content of the artifact",
                    "lifespan": "New lifespan of the artifact",
                },
            },
        }

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        return f"Artifact {self.name} content: {self.payload}"

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        self.users[agent_name].add(timestamp)
        if action not in self.actions:
            return f"Unknown action: {action} - Available actions: {self.actions}"
        if action == f"modify_artifact_{self.name}":
            valid, error_message = self.verify_payload(params.get("payload", ""))
            if not valid:
                return f"Failed to modify artifact {self.name}: {error_message}"

            # Do not change the name. This ensure uniqueness of the artifacts
            past_version = {
                "payload": self.payload,
                "lifespan": "inf" if self.lifespan == np.inf else self.lifespan,
                "name": self.name,
                "version": self.version,
                "version_creation_time": self.version_creation_time,
            }
            self.past_versions.append(past_version)

            self.version += 1
            self.version_creation_time = timestamp
            self.payload = params.get("payload", "")
            self.lifespan = params.get("lifespan", self.lifespan)
            self.remaining_time = np.inf if self.lifespan == -1 else self.lifespan
            return f"Artifact {self.name} updated"
        if action == f"destroy_artifact_{self.name}":
            self.remaining_time = 0
            return f"Artifact {self.name} destroyed"
        return ""

    def verify_payload(self, payload) -> Tuple[bool, str]:
        payload = str(payload)
        if not isinstance(payload, str):
            return False, "Payload must be a string for TextArtifact"
        token_count = len(self.payload_encoder.encode(payload))
        if token_count > MAX_TEXT_ARTIFACT_SIZE:
            return (
                False,
                f"Payload exceeds maximum token limit of {MAX_TEXT_ARTIFACT_SIZE} tokens (got {token_count} tokens)",
            )
        return True, ""
