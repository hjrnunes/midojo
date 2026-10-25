"""midojo-run when a task pair fails: the pair is recorded as failed, the run goes on, and results.json is written.

The control plane runs in this process and the agent is scripted. No model and no network.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from click.testing import CliRunner
from rich.console import Console

from midojo import orchestrator
from midojo.agent_client import AgentClient
from midojo.control_plane.config import AppConfig
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.service import ControlPlane
from midojo.orchestrator import BenchmarkOutcome, exit_status, run_benchmark
from midojo.runtimes import UnmanagedRuntime

# The keys a results.json had before failed pairs were recorded: a run with no failed pair has exactly these.
RESULT_KEYS = {
    "run_id",
    "suite_name",
    "utility",
    "utility_reason",
    "security",
    "security_reason",
    "grade_validity",
    "evidence_status",
    "reachability",
    "aggregates",
}
PER_PAIR = ["utility", "utility_reason", "security", "security_reason", "grade_validity", "evidence_status"]
USER_TASKS = ["weather_new_york", "warmest_city", "trip_weather"]


class ScriptedAgent(AgentClient):
    """Answers every prompt with New York's weather, and raises ``raises`` for a prompt containing ``failing``."""

    def __init__(self, failing: str | None = None, raises: BaseException | None = None) -> None:
        self.failing = failing
        self.raises = raises or RuntimeError("model unreachable")
        self.prompts: list[str] = []

    async def send_task(self, prompt, *, session_token=None, tool_endpoint=None):
        self.prompts.append(prompt)
        if self.failing is not None and self.failing in prompt:
            raise self.raises
        return "New York is 72°F and sunny"


@pytest.fixture
def unmanaged_suite(suite, monkeypatch):
    monkeypatch.setattr(suite, "new_runtime", UnmanagedRuntime)
    return suite


@pytest.fixture
def control(unmanaged_suite) -> ControlPlane:
    return ControlPlane({"weather": unmanaged_suite}, config=AppConfig(closure_wait_seconds=0))


@pytest.fixture
def output(monkeypatch) -> io.StringIO:
    """What the orchestrator prints, as plain text."""
    text = io.StringIO()
    monkeypatch.setattr(orchestrator, "console", Console(file=text, width=200, no_color=True))
    return text


async def benchmark(control, suite, agent, tmp_path, user_tasks, injection_tasks=(), control_client=None):
    async with (control_client or InProcessControlClient)(control) as in_process:
        return await run_benchmark(
            control_url="http://control",
            agent_client=agent,
            agent_uri="http://agent",
            protocol="http",
            suite=suite,
            suite_name="weather",
            user_task_ids=list(user_tasks),
            injection_task_ids=list(injection_tasks),
            logdir=tmp_path,
            control=in_process,
            runtime=UnmanagedRuntime(),
        )


def read_results(tmp_path) -> dict:
    return json.loads((tmp_path / "results.json").read_text())


@pytest.mark.asyncio
async def test_a_failed_pair_is_recorded_and_the_later_pairs_still_run(control, unmanaged_suite, tmp_path, output):
    agent = ScriptedAgent(failing="warmest weather")
    outcome = await benchmark(control, unmanaged_suite, agent, tmp_path, USER_TASKS)

    assert (outcome.graded, outcome.failed) == (2, 1)
    assert exit_status(outcome) == 3
    assert len(agent.prompts) == 3  # the pair after the failed one ran too
    [run] = control.store.list_runs()
    evaluations = {e.user_task_id: e for e in control.evaluations(run.id)}
    failed = evaluations["warmest_city"]
    # The episode context left the failed pair's evaluation sealed and ungraded, with incomplete evidence.
    assert (failed.stage.value, failed.graded) == ("sealed", False)
    assert failed.snapshot is not None and failed.snapshot.evidence_status["agent_output"] == "incomplete"

    results = read_results(tmp_path)
    assert set(results) == RESULT_KEYS | {"status", "failure_reason", "evaluation_id"}
    pairs = ["weather_new_york,", "warmest_city,", "trip_weather,"]
    assert list(results["status"]) == pairs
    assert results["status"] == {"weather_new_york,": "graded", "warmest_city,": "failed", "trip_weather,": "graded"}
    assert results["failure_reason"] == {
        "weather_new_york,": None,
        "warmest_city,": "RuntimeError: model unreachable",
        "trip_weather,": None,
    }
    assert results["evaluation_id"] == {f"{task},": evaluations[task].id for task in USER_TASKS}
    for key in [*PER_PAIR, "reachability"]:
        assert list(results[key]) == pairs
        assert results[key]["warmest_city,"] is None
    assert results["utility"] == {"weather_new_york,": True, "warmest_city,": None, "trip_weather,": False}
    # The failed pair is in neither rate: it was never graded, so it's neither valid nor invalid.
    assert results["aggregates"]["utility"] == {
        "graded": 2,
        "valid": 2,
        "invalid": 0,
        "true": 1,
        "false": 1,
        "rate": 0.5,
    }
    assert results["aggregates"]["failed"] == 1

    printed = output.getvalue()
    assert f"[eval: {failed.id}] warmest_city" in printed
    assert "💥 failed (not graded)   (RuntimeError: model unreachable)" in printed
    assert "Task pairs: 2 graded, 1 failed" in printed


@pytest.mark.asyncio
async def test_a_failed_injection_pair_is_null_in_every_per_pair_map(control, unmanaged_suite, tmp_path):
    # Only the prompt injection changes the prompt: that pair fails, the other is graded.
    agent = ScriptedAgent(failing="By the way")
    injections = ["tornado_alert_via_prompt", "hate_humans_output_hijack"]
    outcome = await benchmark(control, unmanaged_suite, agent, tmp_path, ["weather_new_york"], injections)

    assert (outcome.graded, outcome.failed) == (1, 1)
    results = read_results(tmp_path)
    failed, graded = "weather_new_york,tornado_alert_via_prompt", "weather_new_york,hate_humans_output_hijack"
    for key in [*PER_PAIR, "reachability"]:
        assert results[key][failed] is None
    assert results["security"][graded] is False
    assert results["reachability"][graded] is not None
    assert results["aggregates"]["security"]["graded"] == 1
    assert sum(results["aggregates"]["reachability"].values()) == 1
    assert results["status"] == {failed: "failed", graded: "graded"}


@pytest.mark.asyncio
async def test_when_every_pair_fails_results_json_is_still_written(control, unmanaged_suite, tmp_path, output):
    agent = ScriptedAgent(failing="")  # every prompt
    outcome = await benchmark(control, unmanaged_suite, agent, tmp_path, USER_TASKS)

    assert (outcome.graded, outcome.failed) == (0, 3)
    assert exit_status(outcome) == 4
    assert outcome.results_file == tmp_path / "results.json"
    results = read_results(tmp_path)
    assert set(results["status"].values()) == {"failed"}
    assert results["aggregates"]["failed"] == 3
    assert results["aggregates"]["utility"]["graded"] == 0
    assert results["aggregates"]["utility"]["rate"] is None
    assert "Task pairs: 0 graded, 3 failed" in output.getvalue()


@pytest.mark.asyncio
async def test_a_pair_that_fails_before_its_evaluation_opens_has_no_evaluation_id(control, unmanaged_suite, tmp_path):
    class RefusingSecond(InProcessControlClient):
        created = 0

        async def create_evaluation(self, *args, **kwargs):
            RefusingSecond.created += 1
            if RefusingSecond.created == 2:
                raise ConnectionError("control plane went away")
            return await super().create_evaluation(*args, **kwargs)

    agent = ScriptedAgent()
    outcome = await benchmark(control, unmanaged_suite, agent, tmp_path, USER_TASKS, control_client=RefusingSecond)

    assert (outcome.graded, outcome.failed) == (2, 1)
    results = read_results(tmp_path)
    assert results["evaluation_id"]["warmest_city,"] is None
    assert results["failure_reason"]["warmest_city,"] == "ConnectionError: control plane went away"
    assert all(results["evaluation_id"][pair] for pair in ("weather_new_york,", "trip_weather,"))


@pytest.mark.asyncio
async def test_a_run_with_no_failed_pair_writes_and_prints_what_it_always_did(
    control, unmanaged_suite, tmp_path, output
):
    outcome = await benchmark(control, unmanaged_suite, ScriptedAgent(), tmp_path, USER_TASKS)

    assert (outcome.graded, outcome.failed) == (3, 0)
    assert exit_status(outcome) == 0
    results = read_results(tmp_path)
    assert set(results) == RESULT_KEYS
    assert set(results["aggregates"]) == {"utility", "security", "reachability"}
    printed = output.getvalue()
    assert "💥" not in printed and "not graded" not in printed
    assert "Task pairs:" not in printed


@pytest.mark.asyncio
async def test_a_cancellation_still_stops_the_run(control, unmanaged_suite, tmp_path):
    agent = ScriptedAgent(failing="warmest weather", raises=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await benchmark(control, unmanaged_suite, agent, tmp_path, USER_TASKS)
    assert len(agent.prompts) == 2  # no later pair ran
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize("exiting", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_still_stops_the_run(control, unmanaged_suite, tmp_path, exiting):
    agent = ScriptedAgent(failing="warmest weather", raises=exiting())
    # An interrupt leaves the event loop, so the run has a loop of its own.
    with pytest.raises(exiting):
        asyncio.run(benchmark(control, unmanaged_suite, agent, tmp_path, USER_TASKS))
    assert len(agent.prompts) == 2
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize(
    "outcome,status",
    [
        (BenchmarkOutcome(graded=3, failed=0), 0),
        (BenchmarkOutcome(graded=0, failed=0), 0),
        (BenchmarkOutcome(graded=2, failed=1), 3),
        (BenchmarkOutcome(graded=0, failed=3), 4),
    ],
)
def test_exit_status(outcome, status):
    assert exit_status(outcome) == status


@pytest.mark.parametrize("failing,expected", [(None, 0), ("warmest weather", 3), ("", 4)])
def test_the_cli_exits_with_the_runs_status_after_writing_results_json(
    control, unmanaged_suite, tmp_path, monkeypatch, failing, expected
):
    real = orchestrator.run_benchmark

    async def in_process(**kwargs):
        # The CLI's run, with the scripted agent and the control plane in this process.
        async with InProcessControlClient(control) as in_process:
            return await real(**{**kwargs, "agent_client": ScriptedAgent(failing), "control": in_process})

    monkeypatch.setattr(orchestrator, "get_suite", lambda _name: unmanaged_suite)
    monkeypatch.setattr(orchestrator, "run_benchmark", in_process)
    args = ["--suite", "weather", "--agent-uri", "http://agent", "--protocol", "http", "--logdir", str(tmp_path)]
    args += ["-ut", "weather_new_york", "-ut", "warmest_city", "-ut", "trip_weather"]
    result = CliRunner().invoke(orchestrator.main, args)
    assert result.exit_code == expected, result.output
    assert (tmp_path / "results.json").exists()
