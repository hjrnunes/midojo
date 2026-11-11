"""A name that moved is not still reachable through the module it left, and a module that moved is gone.

A breaking move has no shim: a caller still importing the old path fails at once, instead of working until the
module stops importing the name for its own use. Each module here reads the names it takes from elsewhere through
their module, so none of them is an attribute of the old one; and the modules the interceptor left are not importable.
"""

from __future__ import annotations

import importlib

import midojo_spec.injection
import midojo_spec.judge
import midojo_spec.link
import pytest

LINK_NAMES = tuple(midojo_spec.link.__all__)
MOVED = [("midojo_mim.link", name, "midojo_spec.link") for name in LINK_NAMES] + [
    ("midojo_mim.mcp", "MIDOJO_MCP_CAPABILITY", "midojo_spec.injection"),
    ("midojo_sdk.mcp", "MIDOJO_MCP_CAPABILITY", "midojo_spec.injection"),
    ("midojo.verifiers.judge", "SCHEMA", "midojo_spec.judge"),
    ("midojo.verifiers.judge", "request_id", "midojo_spec.judge"),
    ("midojo.verifiers.judge", "response_problem", "midojo_spec.judge"),
]

GONE = [
    ("midojo_sdk.link", "midojo_mim.link"),
    ("midojo_sdk.placement", "midojo_mim.placement"),
    ("midojo_sdk.context", "midojo_spec.context"),
    ("midojo.interception_service", "midojo_mim.service"),
]


def test_every_moved_name_is_a_name_of_its_new_home():
    for _, name, home in MOVED:
        assert hasattr(importlib.import_module(home), name), (name, home)


@pytest.mark.parametrize(("old", "name", "home"), MOVED, ids=[f"{old}.{name}" for old, name, _ in MOVED])
def test_a_moved_name_is_not_an_attribute_of_the_module_it_left(old: str, name: str, home: str):
    assert not hasattr(importlib.import_module(old), name), f"{old}.{name} still resolves; it is {home}.{name}"


@pytest.mark.parametrize(("old", "home"), GONE, ids=[old for old, _ in GONE])
def test_a_module_that_moved_cannot_be_imported(old: str, home: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)
    importlib.import_module(home)
