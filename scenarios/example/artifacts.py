"""The artifact type of the example scenario: a shelter that keeps a cell safe from storms."""

from typing import Tuple

from terralingua.environment.artifact import Artifact, register_artifact_type


@register_artifact_type("shelter")
class ShelterArtifact(Artifact):
    """A fixed shelter. Beings on its cell take no storm damage."""

    description = "A shelter. Beings on its cell are safe from storms."

    def __init__(self, *args, **kwargs):
        kwargs["movable"] = False
        super().__init__(*args, **kwargs)

    @property
    def actions(self) -> dict:
        return {}

    def interact(self, agent_name: str, action: str, params: dict, timestamp: int) -> str:
        return ""

    def passive_effect(self, timestamp: int, agent_name: str) -> str:
        return "A shelter stands here. Beings on this cell are safe from storms."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""
