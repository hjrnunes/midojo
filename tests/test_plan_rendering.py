"""Compiling a suite's injections into a plan renders every bundled suite pair exactly as before.

``tests/fixtures/plan_rendering.json`` holds, for each bundled suite pair, the
SHA-256 of what the episode delivered before injections became a plan: its
environment, its turns, and each seeded file with the active payloads
substituted. It was written by :func:`renderings` at the commit before the
plan existed (``python -m tests.test_plan_rendering`` writes it again).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from midojo.episode import thaw
from midojo.probes import substitute_probes, substitute_probes_deep
from midojo.suites import get_suite, list_suites
from midojo.yaml_task_suite import YAMLTaskSuite

FIXTURE = Path(__file__).parent / "fixtures" / "plan_rendering.json"


def _pairs(suite: YAMLTaskSuite) -> list[tuple[str, str | None]]:
    injections: list[str | None] = [None, *suite.injection_tasks]
    return [(user, injection) for user in suite.user_tasks for injection in injections]


# Seeded files a suite no longer has, with the template the fixture was written with. They carried no probe, so
# every pair still renders as before with them put back. The weather suite seeded a project .pi/mcp.json that
# pointed the agent at a fake MCP server in its sandbox, until its interception moved to the host.
REMOVED_FILES: dict[str, dict[str, str]] = {
    "weather": {
        ".pi/mcp.json": '{"mcpServers": {"alerts": {"url": "http://localhost:8765/mcp", "exposure": "direct"}}}\n'
    },
}


def _files(suite: YAMLTaskSuite) -> dict[str, str]:
    files = dict(getattr(suite.definition.agent_runtime, "files", {}) or {})
    return {**REMOVED_FILES.get(suite.name, {}), **files}


def _rendering(suite: YAMLTaskSuite, user: str, injection: str | None) -> dict[str, Any]:
    """What one pair's episode delivers: its environment, turns and seeded files, as plain data."""
    probes = suite.get_probes_for_task(injection) if injection is not None else {}
    episode = suite.episode(user, injection, probes)
    active = dict(episode.injections)
    return {
        "environment": thaw(episode.environment),
        "turns": list(episode.turns or ()),
        "files": {path: substitute_probes(template, active) for path, template in _files(suite).items()},
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def renderings() -> dict[str, str]:
    """The digest of every bundled suite pair's rendering, keyed ``"<suite>/<user task>/<injection task>"``."""
    digests = {}
    for name in list_suites():
        suite = get_suite(name)
        for user, injection in _pairs(suite):
            digests[f"{name}/{user}/{injection or '-'}"] = _digest(_rendering(suite, user, injection))
    return digests


def test_every_bundled_pair_renders_as_it_did_before_the_plan():
    expected = json.loads(FIXTURE.read_text())
    assert renderings() == expected


@pytest.mark.parametrize("name", list_suites())
def test_the_plan_renders_like_substituting_the_probes_directly(name):
    # The substitution the suite performed before the plan, applied by hand.
    suite = get_suite(name)
    for user, injection in _pairs(suite):
        probes = suite.get_probes_for_task(injection) if injection is not None else {}
        episode = suite.episode(user, injection, probes)
        assert thaw(episode.environment) == substitute_probes_deep(suite.definition.environment, probes)
        assert list(episode.turns or ()) == [substitute_probes(m, probes) for m in suite.user_tasks[user].turns]
        assert dict(episode.injections) == probes


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(renderings(), indent=2, sort_keys=True) + "\n")
    sys.stdout.write(f"wrote {FIXTURE}\n")
