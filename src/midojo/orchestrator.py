from __future__ import annotations

import asyncio
import contextlib
import importlib
import ipaddress
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from midojo.agent_client import (
    A2AAgentClient,
    AgentClient,
    OGXResponsesClient,
    OpenAIResponsesAgentClient,
    OpenShellAgentClient,
    PIAgentClient,
    SimpleHTTPAgentClient,
)
from midojo.control_plane.client import ControlClient, ControlPlaneClient
from midojo.control_plane.config import CONTROL_TOKEN_ENV
from midojo.control_plane.lifecycle import RejectedError
from midojo.control_plane.models import GradeResponse
from midojo.episode import Episode
from midojo.episode_context import EpisodeContext, EpisodeStep, open_episode
from midojo.reachability import UNKNOWN as UNKNOWN_REACHABILITY
from midojo.reachability import Reachability, count, reachability
from midojo.runtimes import AgentRuntime
from midojo.runtimes.openshell import SANDBOX_HOST, OpenShellRuntime
from midojo.suites import get_suite, list_suites
from midojo.tool_endpoint import bare_host, tool_url
from midojo.yaml_task_suite import YAMLTaskSuite

if TYPE_CHECKING:
    from midojo.interception_host import HostInterception

console = Console()

# What a control client raises when the control plane refuses a request or can't be reached: an
# ``httpx`` error over HTTP, the control plane's own rejection in process.
_CONTROL_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, RejectedError)


def _resolve_system_message(suite_name: str) -> str:
    module_path = suite_name if "." in suite_name else f"suites.{suite_name}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        mod = None
    msg = getattr(mod, "SYSTEM_MESSAGE", None)
    if not msg:
        console.print(f"[dim]No SYSTEM_MESSAGE in '{suite_name}' — running without a system prompt.[/dim]")
        return ""
    return msg


def _finite_seconds(_ctx: click.Context, param: click.Parameter, value: float | None) -> float | None:
    # FloatRange lets nan and inf through; the sandbox exec needs whole seconds.
    if value is not None and not math.isfinite(value):
        raise click.BadParameter(f"{value} is not a finite number of seconds.", param=param)
    return value


class TaskPair(NamedTuple):
    user_task_id: str
    injection_task_id: str


class Exchange(NamedTuple):
    message: str
    reply: str


class TaskResult(NamedTuple):
    eval_id: str
    prompt: str
    agent_output: str
    grade: GradeResponse
    turns: list[Exchange]
    # The sealed snapshot the grade is of.
    snapshot_id: str | None = None


class PairFailure(NamedTuple):
    """A task pair whose ``run_task`` raised: its evaluation, when one was opened, and why it failed.

    In ``results.json`` the pair is null in the grade and evidence maps, left
    out of the utility, security and reachability aggregates and of every
    rate, and counted separately in ``aggregates.failed``. Its evaluation is
    usually sealed ungraded with incomplete evidence, but ``run_task`` can
    raise after the control plane graded it (a lost grade response, say).
    """

    eval_id: str | None
    reason: str


class BenchmarkOutcome(NamedTuple):
    """How many task pairs a run counted (graded) and how many failed, and where its results were written."""

    graded: int
    failed: int
    results_file: Path | None = None


# midojo-run's exit status when task pairs failed: 1 is an unhandled error, 2 a usage error.
EXIT_SOME_FAILED = 3
EXIT_ALL_FAILED = 4


def exit_status(outcome: BenchmarkOutcome) -> int:
    """0 when no pair failed, 3 when some failed and at least one was graded, 4 when every pair failed."""
    if not outcome.failed:
        return 0
    return EXIT_SOME_FAILED if outcome.graded else EXIT_ALL_FAILED


_UNKNOWN = Text("❓ unknown (grade invalid)", style="bold yellow")


def _utility(value: bool | None) -> Text:
    if value is None:
        return _UNKNOWN
    return Text("✅ task completed", style="bold green") if value else Text("❌ task not completed", style="bold red")


_FAILED = Text("💥 failed (not counted)", style="bold magenta")


def _print_agent_text(label: str, text: str, max_len: int = 200) -> None:
    text = (text or "").replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[:max_len] + "…"
    line = Text(f"    {label}: ", style="dim")
    line.append(text, style="italic")
    console.print(line)


def _security(value: bool | None) -> Text:
    if value is None:
        return _UNKNOWN
    if value:
        return Text("💀 attack succeeded", style="bold red")
    return Text("🛡️ attack failed", style="bold green")


def _print_banner(
    suite_name: str,
    agent_uri: str,
    protocol: str,
    user_tasks_to_run: list[str],
    injection_tasks_to_run: list[str],
) -> None:
    lines = Text()
    lines.append("Suite       ", style="dim")
    lines.append(f"{suite_name}\n")
    lines.append("Agent       ", style="dim")
    lines.append(f"{agent_uri} ({protocol})\n")
    lines.append("Tasks       ", style="dim")
    if injection_tasks_to_run:
        lines.append(f"{len(user_tasks_to_run)} user x {len(injection_tasks_to_run)} injection\n")
    else:
        lines.append(f"{len(user_tasks_to_run)} user (no injections)\n")
    console.print(Panel(lines, title="midojo orchestrator", border_style="cyan", padding=(1, 2)))
    console.print()


def _aggregate(results: dict[TaskPair, bool | None]) -> dict[str, int | float | None]:
    """Counts of graded results, and the rate of True among the valid ones.

    An invalid (unknown) grade is counted but left out of the rate's
    denominator; ``rate`` is None when no grade is valid.
    """
    values = list(results.values())
    valid = [v for v in values if v is not None]
    true = sum(1 for v in valid if v)
    return {
        "graded": len(values),
        "valid": len(valid),
        "invalid": len(values) - len(valid),
        "true": true,
        "false": len(valid) - true,
        "rate": true / len(valid) if valid else None,
    }


def _rate_cell(results: dict[TaskPair, bool | None]) -> str:
    aggregate = _aggregate(results)
    rate = aggregate["rate"]
    cell = "-" if rate is None else f"{rate * 100:.1f}%"
    if aggregate["invalid"]:
        cell += f" ({aggregate['invalid']} invalid)"
    return cell


def _print_results_table(
    utility_results: dict[TaskPair, bool | None],
    security_results: dict[TaskPair, bool | None],
    reachability_results: dict[TaskPair, Reachability],
    has_injections: bool,
    results_file: Path,
    pairs: list[TaskPair] | None = None,
    failures: dict[TaskPair, PairFailure] | None = None,
) -> None:
    """The run's results; ``pairs`` orders graded and ``failures`` pairs alike, by default the graded ones."""
    failures = failures or {}
    table = Table(title="Results", border_style="cyan", show_lines=True)
    table.add_column("User Task", style="bold")
    if has_injections:
        table.add_column("Injection Task")
    table.add_column("Utility", justify="center")
    if has_injections:
        table.add_column("Security", justify="center")

    for pair in pairs if pairs is not None else utility_results:
        if pair in failures:
            cells = [_FAILED, _FAILED] if has_injections else [_FAILED]
        elif has_injections:
            cells = [_utility(utility_results[pair]), _security(security_results[pair])]
        else:
            cells = [_utility(utility_results[pair])]
        ids = [pair.user_task_id, pair.injection_task_id] if has_injections else [pair.user_task_id]
        table.add_row(*ids, *cells)

    table.add_section()
    util_avg = _rate_cell(utility_results)
    sec_avg = _rate_cell(security_results)

    if has_injections:
        table.add_row("", "", Text(util_avg, style="bold"), Text(sec_avg, style="bold"))
    else:
        table.add_row("", Text(util_avg, style="bold"))

    console.print(table)
    if has_injections:
        counts = count(reachability_results.values())
        console.print(
            f"\nPayload reachability (report only, not in the rates): {counts['matched']} matched, "
            f"{counts['not_matched']} not matched, {counts['unknown']} unknown"
        )
    if failures:
        console.print(
            f"\nTask pairs: {len(utility_results)} graded, {len(failures)} failed "
            "(not in the utility, security or reachability aggregates or any rate; see failure_reason)"
        )
    console.print(f"\nResults saved to [cyan]{results_file}[/cyan]")


def _turn_label(label: str, idx: int, count: int) -> str:
    """Name a turn's message or reply; a single-turn task keeps the plain label."""
    return f"turn {idx + 1} {label}" if count > 1 else label


def _reachability_detail(report: Reachability) -> str:
    if report.status == "matched":
        return f"payload matched in {report.boundary} ({report.method})"
    if report.status == "not_matched":
        return "payload not matched"
    return "payload match unknown"


async def _injection_reachability(
    control: ControlClient, run_id: str, eval_id: str, injections: dict[str, str]
) -> Reachability:
    """Search every turn's user message and the recorded tool results for the injection payloads.

    Report-only: unknown when the evaluation record can't be fetched or parsed.
    """
    try:
        evaluation = await control.evaluation(run_id, eval_id)
    except (*_CONTROL_ERRORS, ValueError):
        # ValueError covers a body that isn't JSON (JSONDecodeError, UnicodeDecodeError)
        # and one that doesn't match EvaluationResponse (pydantic's ValidationError).
        return UNKNOWN_REACHABILITY
    # A record without started turns falls back to the evaluation's agent input.
    messages = [turn.message for turn in evaluation.turns] or [evaluation.agent_input or ""]
    return reachability(
        injections.values(),
        messages,
        # an unfinished call has no result: the agent saw nothing from it
        [call.result for call in evaluation.function_calls if call.result is not None],
        evaluation.evidence_status.get("function_calls"),
    )


def _check_turns_supported(
    suite: YAMLTaskSuite, user_task_ids: list[str], agent_client: AgentClient, protocol: str
) -> None:
    """Refuse multi-turn user tasks the agent client can't hold a conversation for."""
    if agent_client.supports_multiturn:
        return
    for user_task_id in user_task_ids:
        task = suite.user_tasks.get(user_task_id)
        if task is not None and len(task.turns) > 1:
            raise click.UsageError(
                f"User task {user_task_id!r} has {len(task.turns)} turns, but --protocol {protocol} sends one "
                "message per evaluation. Only --protocol openai and the openshell runtime support more than one turn."
            )


async def run_task(
    control: ControlClient,
    agent_client: AgentClient,
    run_id: str,
    episode: Episode,
    *,
    runtime: AgentRuntime,
    interception: HostInterception | None = None,
    on_context: Callable[[EpisodeContext], object] | None = None,
) -> TaskResult:
    """Run one evaluation of a suite's ``episode`` in an episode context, and grade it on the control plane.

    The context (:func:`midojo.episode_context.open_episode`) opens the
    evaluation in the run (the control plane compiles the same episode from the
    run's suite, and its turns are the ones sent), binds interception and sets
    the runtime up. Each turn is started, sent to the agent and its reply
    recorded, all in one conversation. Then the context closes the evaluation
    (calls already admitted may finish), records the runtime's final
    observations, tears the runtime down and seals, and the control plane
    grades the snapshot.

    Without ``interception``, the runtime and the agent get the evaluation's
    session, for an interception layer in the agent. With it (host mode), the
    evaluation is opened for host interception, and an interceptor started for
    it with the session's recorder and the episode's tool-output plan: the
    runtime and the agent get only its tool endpoint. It stops once the
    evaluation is sealed and graded, or the evaluation fails; calls it gets
    once the evaluation is closing, and after the seal, are late facts.

    ``on_context`` is given the episode context before it's entered, so a
    caller can still read its ``evaluation_id`` (once opened) when this raises.
    """
    user_task_id, injection_task_id = episode.user_task_id, episode.injection_task_id
    if user_task_id is None:
        raise ValueError("run_task runs a suite's episode: compile one with suite.episode()")
    pair = f"{user_task_id} x {injection_task_id}" if injection_task_id else user_task_id
    steps: dict[EpisodeStep, str] = {
        "interceptor": "starting the interceptor",
        "setup": f"setting up the {runtime.name} runtime",
        "close": "waiting for unfinished tool calls",
        "observe": "collecting runtime observations",
        "teardown": f"tearing down the {runtime.name} runtime",
    }

    def progress(step: EpisodeStep) -> contextlib.AbstractContextManager[object]:
        if step not in steps:
            return contextlib.nullcontext()
        return console.status(f"[dim]{pair} · {steps[step]}…[/dim]", spinner="dots")

    context = open_episode(
        episode, runtime=runtime, interception=interception, control=control, run_id=run_id, progress=progress
    )
    if on_context is not None:
        on_context(context)
    async with context as ep:
        turns = await _converse(ep, agent_client, pair)
        agent_output = turns[-1].reply
        snapshot = await ep.seal(agent_output)
        # Graded while the interceptor is still up: a call the target makes after the seal is a late fact.
        grade = await control.grade_evaluation(run_id, ep.evaluation_id)
    return TaskResult(
        eval_id=ep.evaluation_id,
        prompt=turns[0].message,
        agent_output=agent_output,
        grade=grade,
        turns=turns,
        snapshot_id=snapshot.snapshot_id,
    )


async def _converse(ep: EpisodeContext, agent_client: AgentClient, pair: str) -> list[Exchange]:
    """Send the episode's turns to the agent, in one conversation, recording each reply."""
    messages = ep.turns
    assert messages is not None  # A suite's episode always has its user task's turns.
    conversation = agent_client.start_conversation(turn_count=len(messages), **ep.binding.access)
    turns: list[Exchange] = []
    try:
        for idx, message in enumerate(messages):
            running = "running the agent" if len(messages) == 1 else f"running turn {idx + 1}/{len(messages)}"
            with console.status(f"[dim]{pair} · {running}…[/dim]", spinner="dots"):
                reply = await ep.turn(message, conversation.send)
            turns.append(Exchange(message, reply))
    finally:
        await conversation.aclose()
    return turns


async def _run_benchmark(
    control: ControlClient,
    control_url: str,
    agent_client: AgentClient,
    agent_uri: str,
    protocol: str,
    suite: YAMLTaskSuite,
    suite_name: str,
    user_task_ids: list[str] | None,
    injection_task_ids: list[str] | None,
    logdir: Path,
    runtime: AgentRuntime,
    interception: HostInterception | None = None,
) -> BenchmarkOutcome:
    user_tasks_to_run = user_task_ids or list(suite.user_tasks.keys())
    injection_tasks_to_run: list[str]
    if injection_task_ids is not None:
        injection_tasks_to_run = injection_task_ids
    elif user_task_ids is None:
        injection_tasks_to_run = list(suite.injection_tasks.keys())
    else:
        # -ut without -it: utility-only run
        injection_tasks_to_run = []

    _check_turns_supported(suite, user_tasks_to_run, agent_client, protocol)
    if interception is not None and not agent_client.supports_host_interception:
        raise click.UsageError(
            f"--protocol {protocol} can't give the agent a tool endpoint. Host interception supports --protocol "
            "openai, ogx and pi, and the openshell runtime."
        )
    run_id = (await control.create_run(suite_name)).id
    _print_banner(suite_name, agent_uri, protocol, user_tasks_to_run, injection_tasks_to_run)
    console.print(f"  [dim]run[/dim] [cyan underline]{run_id}[/cyan underline]\n")

    utility_results: dict[TaskPair, bool | None] = {}
    utility_reasons: dict[TaskPair, str | None] = {}
    security_results: dict[TaskPair, bool | None] = {}
    security_reasons: dict[TaskPair, str | None] = {}
    grade_validity: dict[TaskPair, dict] = {}
    evidence_status: dict[TaskPair, dict[str, str]] = {}
    reachability_results: dict[TaskPair, Reachability] = {}
    pairs: list[TaskPair] = []
    eval_ids: dict[TaskPair, str | None] = {}
    failures: dict[TaskPair, PairFailure] = {}

    it_ids_to_run: list[str | None] = [*injection_tasks_to_run] if injection_tasks_to_run else [None]
    started: str | None = None
    try:
        with console.status(f"[dim]starting the {runtime.name} runtime…[/dim]", spinner="dots"):
            started = await asyncio.to_thread(runtime.start_run, run_id, suite_name=suite_name)
        if started:
            console.print(f"  [magenta]{runtime.name}[/magenta] [dim]{started}[/dim] [green]ready[/green]\n")

        for ut_id in user_tasks_to_run:
            for it_id in it_ids_to_run:
                episode = suite.episode(ut_id, it_id, suite.get_probes_for_task(it_id) if it_id else {})
                pair = TaskPair(ut_id, it_id or "")
                pairs.append(pair)
                pair_name = f"{ut_id} x {it_id}" if it_id else ut_id
                label = f"[bold]{ut_id}[/bold] x [bold]{it_id}[/bold]" if it_id else f"[bold]{ut_id}[/bold]"
                contexts: list[EpisodeContext] = []
                cancellations = _cancellations()
                try:
                    result = await run_task(
                        control,
                        agent_client,
                        run_id,
                        episode,
                        runtime=runtime,
                        interception=interception,
                        on_context=contexts.append,
                    )
                except Exception as error:
                    # An interrupt or a cancellation isn't an Exception, and stops the run. But the episode
                    # context raises its own failure ahead of a cancellation that arrived while it released
                    # the evaluation, noting the cancellation only: the task's cancellation count went up then.
                    # Only a rise during this pair counts: one a caller suppressed earlier isn't this pair's.
                    if _cancellations() > cancellations:
                        raise asyncio.CancelledError(f"Cancelled while running {pair_name}") from error
                    # The pair failed. Its evaluation, if opened, is sealed and usually ungraded, but
                    # run_task can raise after the grade: either way the pair isn't counted.
                    failure = PairFailure(_opened_evaluation(contexts), f"{type(error).__name__}: {error}")
                    failures[pair] = failure
                    eval_ids[pair] = failure.eval_id
                    _print_failed_pair(control_url, run_id, label, failure)
                    continue
                grade = result.grade
                utility_results[pair] = grade.utility
                utility_reasons[pair] = grade.utility_reason
                grade_validity[pair] = grade.grade_validity.model_dump(mode="json")
                evidence_status[pair] = dict(grade.evidence_status)
                eval_id = eval_ids[pair] = result.eval_id
                eval_url = f"{control_url}/runs/{run_id}/evaluations/{eval_id}"
                console.print(f"  [dim]\\[eval: [link={eval_url}][cyan]{eval_id}[/cyan][/link]][/dim] {label}")
                for idx, (message, reply) in enumerate(result.turns):
                    _print_agent_text(_turn_label("agent input", idx, len(result.turns)), message)
                    _print_agent_text(_turn_label("agent output", idx, len(result.turns)), reply)
                console.print("    ", _utility(grade.utility))
                if it_id:
                    # Every injection evaluation counts in the attack-success rate. Whether the
                    # payload text was found is reported alongside and never filters it out.
                    security_results[pair] = grade.security
                    security_reasons[pair] = grade.security_reason
                    reachability_results[pair] = await _injection_reachability(
                        control, run_id, eval_id, dict(episode.injections)
                    )
                    detail = _reachability_detail(reachability_results[pair])
                    if grade.security is not False and grade.security_reason:
                        # attack succeeded, or couldn't be graded — name the criterion that decided it
                        detail += f" · {grade.security_reason}"
                    console.print("    ", _security(grade.security), Text(f"  ({detail})", style="dim"))
    finally:
        with console.status(f"[dim]stopping the {runtime.name} runtime…[/dim]", spinner="dots"):
            await asyncio.to_thread(runtime.end_run)
        if started:
            console.print(f"\n  [magenta]{runtime.name}[/magenta] [dim]{started}[/dim] [green]cleaned up[/green]")

    console.print()

    logdir.mkdir(parents=True, exist_ok=True)
    results_file = logdir / "results.json"

    def by_pair(values: dict) -> dict:
        return {f"{k.user_task_id},{k.injection_task_id}": values.get(k) for k in pairs}

    results: dict[str, object] = {
        "run_id": run_id,
        "suite_name": suite_name,
        # A failed pair is null in each of these, whatever the control plane holds for its evaluation.
        "utility": by_pair(utility_results),
        "utility_reason": by_pair(utility_reasons),
        # null without an injection task, or where the grade is invalid
        "security": by_pair(security_results),
        "security_reason": by_pair(security_reasons),
        "grade_validity": by_pair(grade_validity),
        "evidence_status": by_pair(evidence_status),
        # Whether the payload text was found; report-only, null without an injection task.
        "reachability": by_pair({k: v.model_dump(mode="json") for k, v in reachability_results.items()}),
    }
    # Rates leave invalid grades out of the denominator and failed pairs out altogether; reachability never enters them.
    aggregates: dict[str, object] = {
        "utility": _aggregate(utility_results),
        "security": _aggregate(security_results),
        "reachability": count(reachability_results.values()),
    }
    if failures:
        # Only a run with a failed pair has these: a run with none writes what it always did.
        results["status"] = by_pair({k: "failed" if k in failures else "graded" for k in pairs})
        results["failure_reason"] = by_pair({k: f.reason for k, f in failures.items()})
        results["evaluation_id"] = by_pair(eval_ids)
        aggregates["failed"] = len(failures)
    results["aggregates"] = aggregates
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    _print_results_table(
        utility_results,
        security_results,
        reachability_results,
        bool(injection_tasks_to_run),
        results_file,
        pairs,
        failures,
    )
    return BenchmarkOutcome(graded=len(utility_results), failed=len(failures), results_file=results_file)


def _cancellations() -> int:
    """The current task's pending cancellation requests (``Task.cancelling``): requested and not withdrawn."""
    task = asyncio.current_task()
    return task.cancelling() if task is not None else 0


def _opened_evaluation(contexts: list[EpisodeContext]) -> str | None:
    """The evaluation the pair's episode context opened, or None when it failed before opening one."""
    try:
        return contexts[0].evaluation_id
    except (IndexError, RuntimeError):
        return None


def _print_failed_pair(control_url: str, run_id: str, label: str, failure: PairFailure) -> None:
    if failure.eval_id is None:
        console.print(f"  [dim]\\[eval: none][/dim] {label}")
    else:
        eval_url = f"{control_url}/runs/{run_id}/evaluations/{failure.eval_id}"
        console.print(f"  [dim]\\[eval: [link={eval_url}][cyan]{failure.eval_id}[/cyan][/link]][/dim] {label}")
    console.print("    ", _FAILED, Text(f"  ({failure.reason})", style="dim"))


async def run_benchmark(
    control_url: str,
    agent_client: AgentClient,
    agent_uri: str,
    protocol: str,
    suite: YAMLTaskSuite,
    suite_name: str,
    user_task_ids: list[str] | None,
    injection_task_ids: list[str] | None,
    logdir: Path,
    control_token: str | None = None,
    http: httpx.AsyncClient | None = None,
    runtime: AgentRuntime | None = None,
    control: ControlClient | None = None,
    interception: HostInterception | None = None,
) -> BenchmarkOutcome:
    """Run the benchmark against the control plane at ``control_url``, and say how many task pairs failed.

    ``http`` is the client for control-plane requests; by default the control
    plane client opens its own. ``control`` replaces that client altogether,
    such as an ``InProcessControlClient`` for a control plane in this process;
    the caller keeps it open and closes it, and ``control_url`` only names the
    evaluations in the output. ``runtime`` runs the run's evaluations, one at
    a time; by default it's a fresh one from the suite (``suite.new_runtime()``).
    ``interception`` runs the evaluations in host mode, each with its own
    interceptor (see :func:`run_task`); by default the agent gets the session.

    A task pair whose evaluation fails (``run_task`` raises an ``Exception``)
    is recorded as failed and the run goes on; ``results.json`` is written
    either way. :func:`exit_status` maps the outcome to midojo-run's exit status.
    """
    if runtime is None:
        runtime = suite.new_runtime()
    async with contextlib.AsyncExitStack() as stack:
        if control is None:
            control = await stack.enter_async_context(ControlPlaneClient(control_url, token=control_token, http=http))
        return await _run_benchmark(
            control,
            control_url,
            agent_client,
            agent_uri,
            protocol,
            suite,
            suite_name,
            user_task_ids,
            injection_task_ids,
            logdir,
            runtime,
            interception,
        )


@click.command()
@click.option(
    "--control-url",
    default="http://localhost:8080",
    help="URL of the control plane started with midojo-serve. "
    "OpenShell sandboxes get this URL too, with localhost rewritten to reach the host.",
)
@click.option(
    "--control-token",
    envvar=CONTROL_TOKEN_ENV,
    default=None,
    help=f"Control token the control plane was started with. Never passed to the agent; env: {CONTROL_TOKEN_ENV}.",
)
@click.option(
    "--gateway",
    default=None,
    help="OpenShell gateway to run the agent's sandboxes on. Required for suites with the openshell agent_runtime.",
)
@click.option(
    "--agent-uri",
    default=None,
    help="Where to reach an unmanaged agent. A URL for http/a2a/ogx/openai; a local path to the agent dir for pi.",
)
@click.option(
    "--suite", "suite_name", required=True, help=f"Benchmark suite name. Built-in: {', '.join(list_suites())}."
)
@click.option("--user-task", "-ut", "user_tasks", multiple=True, default=(), help="Specific user task IDs.")
@click.option(
    "--injection-task", "-it", "injection_tasks", multiple=True, default=(), help="Specific injection task IDs."
)
@click.option("--logdir", default="./runs", type=Path, help="Directory to store results.")
@click.option(
    "--module-to-load", "-ml", "modules_to_load", multiple=True, default=(), help="Additional modules to import."
)
@click.option(
    "--protocol",
    type=click.Choice(["http", "a2a", "pi", "ogx", "openai"]),
    default=None,
    help="How to talk to an unmanaged agent. "
    "API keys are read from env vars: OPENAI_API_KEY (openai), OGX_CLIENT_API_KEY (ogx).",
)
@click.option(
    "--ogx-shield", default=None, envvar="OGX_SHIELD_ID", help="Shield ID for OGX guardrails (ogx protocol only)."
)
@click.option(
    "--mcp-server-url",
    envvar="MCP_SERVER_URL",
    help="MCP server URL reachable by the inference server. Required for ogx/openai; env: MCP_SERVER_URL.",
)
@click.option(
    "--mcp-server-label",
    default=None,
    envvar="MCP_SERVER_LABEL",
    help="Label the MCP server is registered under on the agent's inference server. "
    "Defaults to the suite name. Override when the server expects a different label.",
)
@click.option(
    "--model-name",
    default=None,
    envvar="MODEL_NAME",
    help="Model ID. Required for ogx/openai; env: MODEL_NAME.",
)
@click.option(
    "--agent-timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    callback=_finite_seconds,
    help="Seconds the agent may take per evaluation. Overrides the suite's timeout_seconds for openshell suites "
    "(default 300); for unmanaged agents, sets the client's timeout (default 120).",
)
@click.option(
    "--interception",
    type=click.Choice(["agent", "host"]),
    default="agent",
    show_default=True,
    help="Where tool calls are intercepted. agent: an SDK in the agent records them, with the evaluation's session "
    "(lower-trust evidence). host: midojo runs an interceptor for each evaluation in this process, and the agent "
    "gets only its tool endpoint, never a session or the control-plane URL.",
)
@click.option(
    "--upstream-mcp-url",
    envvar="UPSTREAM_MCP_URL",
    default=None,
    help="The real MCP server the host-side interceptor mirrors and forwards to. Required with --interception host; "
    "env: UPSTREAM_MCP_URL.",
)
@click.option(
    "--interceptor-bind",
    default="127.0.0.1",
    show_default=True,
    help="Address the host-side interceptor listens on. An OpenShell sandbox can't reach the host's loopback: bind "
    "an address it reaches (e.g. 0.0.0.0).",
)
@click.option(
    "--interceptor-port",
    type=click.IntRange(0, 65535),
    default=0,
    show_default=True,
    help="Port the host-side interceptor listens on; 0 picks a free one per evaluation. OpenShell needs a fixed "
    "port, the one its network policy allows.",
)
@click.option(
    "--interceptor-advertise-host",
    default=None,
    help="Host the agent connects to the interceptor at. Defaults to host.openshell.internal for OpenShell "
    "sandboxes, and to the bind address otherwise (127.0.0.1 for a wildcard bind).",
)
def main(
    control_url: str,
    control_token: str | None,
    gateway: str | None,
    agent_uri: str | None,
    suite_name: str,
    user_tasks: tuple[str, ...],
    injection_tasks: tuple[str, ...],
    logdir: Path,
    modules_to_load: tuple[str, ...],
    protocol: str | None,
    ogx_shield: str | None,
    mcp_server_url: str | None,
    mcp_server_label: str | None,
    model_name: str | None,
    agent_timeout: float | None,
    interception: str,
    upstream_mcp_url: str | None,
    interceptor_bind: str,
    interceptor_port: int,
    interceptor_advertise_host: str | None,
) -> None:
    """Run a benchmark suite's task pairs against an agent, and write their results to LOGDIR/results.json.

    A task pair whose evaluation fails is recorded as failed, and the run goes on.

    \b
    Exit status:
      0  no task pair failed
      3  some task pairs failed, and at least one was graded
      4  every task pair failed
    """
    for module in modules_to_load:
        importlib.import_module(module)

    suite = get_suite(suite_name)
    # A fresh runtime for this run, from the suite's agent_runtime: the suite keeps none.
    runtime = suite.new_runtime()
    agent_client: AgentClient
    # Unmanaged clients keep their own default timeout unless --agent-timeout is given.
    timeout_kwargs = {} if agent_timeout is None else {"timeout": agent_timeout}
    host_mode = interception == "host"
    host_interception = (
        _host_interception(
            upstream_mcp_url,
            bind=interceptor_bind,
            port=interceptor_port,
            advertise=interceptor_advertise_host,
            openshell=isinstance(runtime, OpenShellRuntime),
        )
        if host_mode
        else None
    )
    if not host_mode and (interceptor_port or interceptor_advertise_host):
        raise click.UsageError("The --interceptor-* options apply only to --interception host.")

    if isinstance(runtime, OpenShellRuntime):
        if agent_uri or protocol:
            raise click.UsageError(
                f"Suite {suite_name!r} runs its agent in OpenShell; pass --gateway instead of --agent-uri/--protocol."
            )
        if not gateway:
            raise click.UsageError(f"Suite {suite_name!r} runs its agent in OpenShell and requires --gateway.")
        if host_interception is not None:
            # The sandbox never gets the control plane's URL in host mode, only the interceptor's.
            assert host_interception.advertise_host is not None
            advertised = tool_url(host_interception.advertise_host, host_interception.port)
            runtime.configure(cluster=gateway, interceptor_url=advertised)
        else:
            runtime.configure(cluster=gateway, control_url=control_url)
        agent_client = OpenShellAgentClient(runtime, timeout=agent_timeout)
        agent_uri, protocol = gateway, "openshell"
    elif gateway:
        raise click.UsageError(
            f"--gateway applies only to suites with the openshell agent_runtime; {suite_name!r} is {runtime.name}."
        )
    elif not agent_uri or not protocol:
        raise click.UsageError(f"Suite {suite_name!r} has an unmanaged agent and requires --agent-uri and --protocol.")
    elif protocol == "a2a":
        agent_client = A2AAgentClient(agent_uri, **timeout_kwargs)
    elif protocol == "pi":
        agent_client = PIAgentClient(agent_uri, control_url, **timeout_kwargs)
    elif protocol in {"ogx", "openai"}:
        if not model_name:
            raise click.UsageError(f"--protocol {protocol} requires --model-name or MODEL_NAME.")
        if not mcp_server_url and not host_mode:
            raise click.UsageError(f"--protocol {protocol} requires --mcp-server-url or MCP_SERVER_URL.")
        system_message = _resolve_system_message(suite_name)
        if protocol == "ogx":
            agent_client = OGXResponsesClient(
                ogx_url=agent_uri,
                model=model_name,
                mcp_server_url=mcp_server_url,
                mcp_server_label=mcp_server_label or suite_name,
                instructions=system_message,
                shield_id=ogx_shield,
                **timeout_kwargs,
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise click.UsageError("--protocol openai requires OPENAI_API_KEY.")
            agent_client = OpenAIResponsesAgentClient(
                base_url=agent_uri,
                model=model_name,
                mcp_server_url=mcp_server_url,
                mcp_server_label=mcp_server_label or suite_name,
                api_key=api_key,
                instructions=system_message,
                **timeout_kwargs,
            )
    else:
        agent_client = SimpleHTTPAgentClient(agent_uri, **timeout_kwargs)

    outcome = asyncio.run(
        run_benchmark(
            control_url=control_url,
            agent_client=agent_client,
            agent_uri=agent_uri,
            protocol=protocol,
            suite=suite,
            suite_name=suite_name,
            user_task_ids=list(user_tasks) if user_tasks else None,
            injection_task_ids=list(injection_tasks) if injection_tasks else None,
            logdir=logdir,
            control_token=control_token or None,
            runtime=runtime,
            interception=host_interception,
        )
    )
    status = exit_status(outcome)
    if status:
        sys.exit(status)


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``host`` as an IP address (an IPv4-mapped IPv6 one as its IPv4 address), or None for a name."""
    try:
        ip = ipaddress.ip_address(bare_host(host))
    except ValueError:
        return None
    return ip.ipv4_mapped or ip if isinstance(ip, ipaddress.IPv6Address) else ip


def _is_loopback(host: str) -> bool:
    """Whether ``host`` is this machine's loopback: ``localhost`` or an address in 127.0.0.0/8 or ::1."""
    name = host.lower().rstrip(".")
    ip = _ip(host)
    return name == "localhost" or name.endswith(".localhost") or (ip is not None and ip.is_loopback)


def _is_unspecified(host: str) -> bool:
    """Whether ``host`` is an unspecified address (0.0.0.0, ::): one to listen on, not to connect to."""
    ip = _ip(host)
    return host == "" or (ip is not None and ip.is_unspecified)


def _host_interception(
    upstream_mcp_url: str | None, *, bind: str, port: int, advertise: str | None, openshell: bool
) -> HostInterception:
    """The run's host-mode settings, from the CLI options, refusing what a target couldn't reach."""
    from midojo.interception_host import HostInterception  # needs midojo-sdk[mcp]

    if not upstream_mcp_url:
        raise click.UsageError("--interception host requires --upstream-mcp-url or UPSTREAM_MCP_URL.")
    # Hosts are kept without URL brackets ([2001:db8::1] is 2001:db8::1), and bracketed once in a URL.
    bind = bare_host(bind)
    advertise = bare_host(advertise) if advertise else None
    if openshell:
        if not port:
            raise click.UsageError(
                "--interception host with the openshell runtime requires --interceptor-port: the sandbox's network "
                "policy must allow a fixed port."
            )
        if _is_loopback(bind):
            raise click.UsageError(
                "An OpenShell sandbox can't reach an interceptor on the host's loopback: pass --interceptor-bind "
                "with an address the sandbox reaches (e.g. 0.0.0.0)."
            )
        advertise = advertise or SANDBOX_HOST
        if _is_loopback(advertise) or _is_unspecified(advertise):
            raise click.UsageError(
                f"An OpenShell sandbox can't reach an interceptor advertised at {advertise}: its loopback is the "
                f"sandbox's own, and an unspecified address names no host. Pass --interceptor-advertise-host with "
                f"a host the sandbox reaches, or leave it out for {SANDBOX_HOST}."
            )
    return HostInterception(upstream=upstream_mcp_url, bind_host=bind, port=port, advertise_host=advertise)


if __name__ == "__main__":
    main()
