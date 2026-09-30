"""A small complete scenario to copy. Select it with run.scenario: scenarios.example."""

from scenarios.example import (
    artifacts,  # noqa: F401  registers the "shelter" artifact type
)
from scenarios.example.storms import Storms, StormsOptions

Options = StormsOptions


def build(options: StormsOptions):
    return [Storms(options)]
