"""A refused artifact edit cannot be remembered as a partially completed write."""

from copy import deepcopy

import numpy as np
import pytest

from terralingua.environment.artifact import TextArtifact


@pytest.fixture
def artifact():
    item = TextArtifact(name="working-plan", payload="Existing guidance", lifespan=20,
                        pose=(0, 0), creator="author", creation_time=1)
    item.remaining_time = 7
    return item


def content_state(item):
    return deepcopy({key: getattr(item, key) for key in (
        "payload", "version", "version_creation_time", "past_versions", "lifespan", "remaining_time"
    )})


@pytest.mark.parametrize("lifespan", ["invalid", None, float("inf")])
def test_invalid_lifespan_leaves_all_content_and_version_fields_unchanged(artifact, lifespan):
    before = content_state(artifact)
    result = artifact.interact("editor", "modify_artifact", {"payload": "Replacement guidance", "lifespan": lifespan}, 9)
    assert result.startswith("Failed to modify")
    assert content_state(artifact) == before
    # An attempted interaction is still an event; it is not a content version.
    assert artifact.users["editor"] == {9}


def test_failed_edit_after_success_preserves_the_last_committed_version(artifact):
    assert artifact.interact("editor", "modify_artifact", {"payload": "First revision", "lifespan": 12}, 8).endswith("updated")
    before = content_state(artifact)
    artifact.interact("editor", "modify_artifact", {"payload": "Uncommitted revision", "lifespan": None}, 9)
    assert content_state(artifact) == before
    assert artifact.past_versions[0]["payload"] == "Existing guidance"


def test_valid_edit_commits_payload_lifetime_and_one_history_version_together(artifact):
    assert artifact.interact("editor", "modify_artifact", {"payload": "Replacement guidance", "lifespan": "-1"}, 9).endswith("updated")
    assert artifact.payload == "Replacement guidance"
    assert artifact.version == 1
    assert artifact.version_creation_time == 9
    assert artifact.lifespan == artifact.remaining_time == np.inf
    assert len(artifact.past_versions) == 1
    previous = artifact.past_versions[0]
    assert (previous["payload"], previous["lifespan"], previous["version"]) == ("Existing guidance", 20, 0)


def test_invalid_payload_cannot_change_lifespan_or_version(artifact):
    artifact.max_tokens = 1
    before = content_state(artifact)
    result = artifact.interact("editor", "modify_artifact", {"payload": "A much longer replacement", "lifespan": -1}, 9)
    assert result.startswith("Failed to modify")
    assert content_state(artifact) == before
