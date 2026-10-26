"""The episode context: one run of an episode, from acquiring its runtime to its sealed snapshot.

:func:`open_episode` takes one resolved :class:`~midojo.episode.Episode` and
returns an async context manager that runs it once::

    async with open_episode(episode, interception=HostInterception(upstream=url)) as ep:
        for message in ep.turns or ():
            await ep.turn(message, lambda m: my_agent(m, ep.binding))
    snapshot = ep.sealed
    verdict = grade(episode, snapshot)

Entering it acquires the runtime, opens the evaluation, binds interception and
sets the runtime up for the agent; the host then drives the agent with
:attr:`EpisodeContext.binding` and reports each turn. Leaving it closes the
evaluation (calls already admitted may finish), collects the runtime's final
observations, tears the runtime down and seals the evaluation. Grading is
separate: :func:`midojo.control_plane.grading.grade` is a pure function of the
episode and the sealed snapshot.

The context never chooses what to send or how to grade: the MiDojo runner
(:func:`midojo.orchestrator.run_task`), an attacker loop, a PyRIT target or a
Garak generator wrapper are all clients that drive one context per episode.
Every result names its evaluation and snapshot; nothing looks up "the latest
evaluation".
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx
from midojo_sdk.injection import Channel, InjectionOutcome

from midojo.control_plane.client import ControlClient
from midojo.control_plane.state import Snapshot
from midojo.episode import DEFAULT_EPISODE_NAME, Episode
from midojo.observations import FUNCTION_CALLS
from midojo.runtimes import AgentRuntime, ResettableRuntime
from midojo.tool_endpoint import ToolEndpoint
from midojo.types import InterceptionMode

if TYPE_CHECKING:
    from midojo.control_plane.models import CreateEvaluationResponse
    from midojo.control_plane.service import ControlPlane
    from midojo.interception_host import HostInterception, HostInterceptor

logger = logging.getLogger(__name__)

# The steps a context takes around the host's turns, in order, for a caller that shows progress.
EpisodeStep = Literal["start_run", "reset", "interceptor", "setup", "close", "observe", "teardown", "end_run"]

# Whether the target was reset before the episode: ``reset`` (asked, and the runtime declares the capability),
# ``not_requested``, or ``unsupported`` (asked, but the runtime declares no reset: the target was not reset, and
# may hold what earlier episodes left).
ResetOutcome = Literal["reset", "not_requested", "unsupported"]

# What the seal of an episode cut short by a failure or a cancellation marks incomplete.
INTERRUPTED_EVIDENCE = (FUNCTION_CALLS, "turns", "agent_output")

T = TypeVar("T")


@dataclass(frozen=True)
class AgentBinding:
    """What the agent under test may hold: the evaluation's session, or in host mode only a tool endpoint.

    Exactly one is set. ``session_token`` is for an interception layer in the
    agent (in-agent mode); ``tool_endpoint`` is the host interceptor's URL and
    bearer token, and grants nothing on the control plane.
    """

    session_token: str | None = field(default=None, repr=False)
    tool_endpoint: ToolEndpoint | None = None

    def __post_init__(self) -> None:
        if (self.session_token is None) == (self.tool_endpoint is None):
            raise ValueError("A binding is exactly one of a session token and a tool endpoint")

    @property
    def mode(self) -> InterceptionMode:
        return "host" if self.tool_endpoint is not None else "agent"

    @property
    def access(self) -> dict[str, Any]:
        """The binding as keyword arguments: what ``AgentRuntime.setup`` and ``AgentClient`` take."""
        if self.tool_endpoint is not None:
            return {"tool_endpoint": self.tool_endpoint}
        return {"session_token": self.session_token}


@dataclass(frozen=True)
class EpisodeResult:
    """What one run of an episode left: its sealed snapshot, and the host-side facts about the run.

    ``injection_outcomes`` are the host interceptor's outcomes for the plan's
    tool-output placements (empty in in-agent mode). ``reset`` says whether the
    target was reset before the episode (see :data:`ResetOutcome`).
    ``interrupted`` is True when a failure or a cancellation cut the episode
    short before the control plane sealed it: it was sealed with what was
    collected, and the snapshot marks :data:`INTERRUPTED_EVIDENCE` incomplete.

    ``snapshot`` is None when the control plane sealed the evaluation but the
    context couldn't read its snapshot back (``ControlClient.snapshot``
    failed): ``snapshot_error`` is that failure. The evaluation is sealed as
    it was, and the snapshot is still on the control plane, under
    ``evaluation_id``; sealing again (:meth:`EpisodeContext.seal`) only
    retries the read.
    """

    evaluation_id: str
    snapshot: Snapshot | None
    interception: InterceptionMode
    reset: ResetOutcome = "not_requested"
    interrupted: bool = False
    injection_outcomes: tuple[InjectionOutcome, ...] = ()
    snapshot_error: BaseException | None = None

    @property
    def snapshot_id(self) -> str | None:
        return self.snapshot.snapshot_id if self.snapshot is not None else None


def _no_progress(_step: EpisodeStep) -> AbstractContextManager[object]:
    return contextlib.nullcontext()


class EpisodeContext:
    """One run of ``episode``: see :func:`open_episode`. It can be entered once."""

    def __init__(
        self,
        episode: Episode,
        *,
        runtime: AgentRuntime | None = None,
        interception: HostInterception | None = None,
        control: ControlPlane | ControlClient | None = None,
        run_id: str | None = None,
        reset: bool = False,
        progress: Callable[[EpisodeStep], AbstractContextManager[object]] | None = None,
    ) -> None:
        self.episode = episode
        # A runtime the caller gives belongs to a run the caller started; one built here is started and ended here.
        self._owns_runtime = runtime is None
        self.runtime: AgentRuntime = runtime if runtime is not None else episode.new_runtime()
        self.interception = interception
        self.run_id = run_id
        self._control_given = control
        self._reset_requested = reset
        self.reset_outcome: ResetOutcome = "not_requested"
        self._progress = progress or _no_progress
        self._entered = False
        self._run_started = False
        self._set_up = False
        self._setup_done = False
        self._torn_down = False
        # Whether a failure or a cancellation cut the episode short of its seal.
        self.interrupted = False
        self._evaluation_id: str | None = None
        self._session_token: str | None = None
        self._binding: AgentBinding | None = None
        self._interceptor: HostInterceptor | None = None
        self._last_reply: str | None = None
        # Whether the control plane has sealed the evaluation, its snapshot read back or not.
        self._control_sealed = False
        # The seal last sent (its agent output and incomplete keys) while its outcome is unknown: sent, and
        # neither accepted nor refused outright.
        self._seal_request: tuple[str, tuple[str, ...]] | None = None
        self._snapshot: Snapshot | None = None
        # Why the sealed snapshot couldn't be read back, when it couldn't.
        self._snapshot_error: BaseException | None = None
        self.turns: tuple[str, ...] | None = None
        self.control: ControlClient

    # --- what the host reads ---

    @property
    def evaluation_id(self) -> str:
        if self._evaluation_id is None:
            raise RuntimeError("The episode's evaluation isn't open yet")
        return self._evaluation_id

    @property
    def binding(self) -> AgentBinding:
        """What to give the agent: its tool endpoint (host mode) or the evaluation's session."""
        if self._binding is None:
            raise RuntimeError("The episode's interception isn't bound yet")
        return self._binding

    @property
    def interceptor(self) -> HostInterceptor | None:
        """The host interceptor serving the binding's tool endpoint, in host mode."""
        return self._interceptor

    @property
    def sealed(self) -> Snapshot:
        """The sealed snapshot: immutable evidence, named by ``evaluation_id`` and ``snapshot_id``."""
        if self._snapshot is None:
            if self._snapshot_error is not None:
                raise RuntimeError(
                    f"Evaluation {self._evaluation_id} is sealed, but its snapshot couldn't be read back"
                ) from self._snapshot_error
            raise RuntimeError("The episode isn't sealed yet")
        return self._snapshot

    @property
    def result(self) -> EpisodeResult:
        """The run's result, once the control plane has sealed the evaluation (its snapshot read back or not)."""
        if self._snapshot is None and not self._control_sealed:
            raise RuntimeError("The episode isn't sealed yet")
        interceptor = self._interceptor
        return EpisodeResult(
            evaluation_id=self.evaluation_id,
            snapshot=self._snapshot,
            interception="host" if self.interception is not None else "agent",
            reset=self.reset_outcome,
            interrupted=self.interrupted,
            injection_outcomes=interceptor.injection_outcomes if interceptor is not None else (),
            snapshot_error=self._snapshot_error,
        )

    # --- entering ---

    async def __aenter__(self) -> EpisodeContext:
        if self._entered:
            raise RuntimeError("An episode context runs its episode once")
        self._entered = True
        self.control = self._resolve_control()
        try:
            await self._acquire()
        except BaseException as failure:
            raised = await self._release_to_the_end(failure)
            if raised is failure or raised is None:
                raise
            raise raised from failure
        return self

    def _resolve_control(self) -> ControlClient:
        from midojo.control_plane.inprocess_control import InProcessControlClient
        from midojo.control_plane.service import ControlPlane

        control = self._control_given
        if control is None:
            # A control plane of its own, in this process: no suite or run registered, no server.
            control = ControlPlane({})
        return InProcessControlClient(control) if isinstance(control, ControlPlane) else control

    async def _acquire(self) -> None:
        episode, runtime = self.episode, self.runtime
        if self._owns_runtime:
            with self._progress("start_run"):
                self._run_started = True
                run_name = self.run_id or f"episode-{uuid.uuid4().hex[:12]}"
                suite_name = episode.suite_name or DEFAULT_EPISODE_NAME
                await _in_thread(runtime.start_run, run_name, suite_name=suite_name)
        if self._reset_requested:
            # Only a declared capability: an external target is never reset behind its owner's back.
            if isinstance(runtime, ResettableRuntime):
                resettable = runtime

                def reset() -> None:
                    resettable.reset()
                    # Set in the step itself, as for setup: a reset that succeeds while a cancellation waits for it
                    # still happened.
                    self.reset_outcome = "reset"

                with self._progress("reset"):
                    await _in_thread(reset)
            else:
                self.reset_outcome = "unsupported"
        await self._open()
        assert self._session_token is not None
        if self.interception is not None:
            with self._progress("interceptor"):
                self._interceptor = self.interception.interceptor(
                    self.control.recorder(self._session_token),
                    evaluation_id=self.evaluation_id,
                    plan=episode.plan.for_channels(Channel.TOOL_OUTPUT),
                )
                self._binding = AgentBinding(tool_endpoint=await self._interceptor.start())
        else:
            self._binding = AgentBinding(session_token=self._session_token)
        binding = self._binding

        def set_up() -> None:
            runtime.setup(
                dict(episode.injections),
                **binding.access,
                eval_id=self.evaluation_id,
                user_task_id=episode.user_task_id,
                injection_task_id=episode.injection_task_id,
            )
            # Set in the step itself: a setup that succeeds while a cancellation waits for it is still done,
            # so the interrupted closure still observes the runtime.
            self._setup_done = True

        with self._progress("setup"):
            self._set_up = True
            await _in_thread(set_up)

    async def _open(self) -> None:
        """Open the evaluation: the episode itself in process, or a suite's episode in a run."""
        from midojo.control_plane.inprocess_control import InProcessControlClient

        episode = self.episode
        if self.run_id is None:
            if not isinstance(self.control, InProcessControlClient):
                raise ValueError("Outside a run, an episode is opened in a control plane in this process")
            mode: InterceptionMode = "host" if self.interception is not None else "agent"
            opened = await self.control.open_episode(episode, interception=mode)
            self._evaluation_id, self._session_token, self.turns = opened.id, opened.session_token, opened.turns
            return
        if episode.user_task_id is None:
            raise ValueError("In a run, the control plane compiles the episode from its suite: it needs a user_task_id")
        # The run's control plane compiles the same suite pair, and its turns are the ones the host sends.
        host: dict[str, InterceptionMode] = {"interception": "host"} if self.interception is not None else {}
        run_id, user_task_id, key = self.run_id, episode.user_task_id, uuid.uuid4().hex

        async def create() -> CreateEvaluationResponse:
            return await self.control.create_evaluation(
                run_id, user_task_id, episode.injection_task_id, dict(episode.injections), idempotency_key=key, **host
            )

        try:
            created = await create()
        except Exception as error:
            # An unknown outcome (its response lost, say): the control plane may have opened the evaluation. The
            # same create, under the same key, is sent again once, and answers with that evaluation if it did.
            if _refused(error):
                raise
            created = await create()
        except BaseException as stopped:
            # A cancellation or an interrupt while waiting for the response: the control plane may have opened the
            # evaluation all the same. The same create is sent again, to its end however often the caller is
            # cancelled meanwhile, so the evaluation it opened is known and cleanup seals it; then this goes on.
            try:
                recovered, _ = await _to_the_end(create())
            except Exception as lost:
                stopped.add_note(
                    f"The evaluation it may have opened couldn't be recovered: {type(lost).__name__}: {lost}"
                )
            else:
                self._opened(recovered)
            raise
        self._opened(created)

    def _opened(self, created: CreateEvaluationResponse) -> None:
        self._evaluation_id, self._session_token, self.turns = created.id, created.session_token, tuple(created.turns)

    # --- turns ---

    async def start_turn(self, message: str) -> int:
        """Start the next turn with the user's ``message``; calls the agent makes from now on are stamped with it."""
        return (await self.control.start_turn(self.run_id, self.evaluation_id, message)).index

    async def record_turn_reply(self, index: int, reply: str) -> None:
        """Record the agent's ``reply`` to turn ``index``. The last reply is the agent's output at the seal."""
        await self.control.record_turn_reply(self.run_id, self.evaluation_id, index, reply)
        self._last_reply = reply

    async def turn(self, message: str, send: Callable[[str], Awaitable[str]]) -> str:
        """Start a turn with ``message``, ``send`` it to the agent, and record and return the reply."""
        index = await self.start_turn(message)
        reply = await send(message)
        await self.record_turn_reply(index, reply)
        return reply

    # --- sealing and releasing ---

    async def seal(self, agent_output: str | None = None) -> Snapshot:
        """Close the evaluation, collect the final observations, tear the runtime down, and seal.

        ``agent_output`` defaults to the last recorded reply. Leaving the
        context seals it if the host hasn't; sealing first lets the host act
        on the sealed evaluation (grade it on the control plane, say) while the
        interceptor is still up, so a call after the seal is a late fact.

        Once the control plane has sealed the evaluation, the snapshot is read
        back (``ControlClient.snapshot``). If that read fails, its error is
        raised, the evaluation stays sealed as it was, and :attr:`result` says
        so (``snapshot`` None, ``snapshot_error``); sealing again only retries
        the read, whatever ``agent_output`` it's given.

        If the seal request itself fails (its response lost, say, or a
        cancellation while waiting for it), the control plane may have sealed
        or not: sealing again sends the same seal again, whatever
        ``agent_output`` it's given. Completing a sealed evaluation with the
        same output is a retry that leaves its snapshot as it was, so the
        evaluation is sealed once either way. Leaving the context after such a
        failure sends it again too (see :meth:`_release`).

        A seal the control plane refuses outright (see :func:`_refused`: an
        invalid output, say) changed neither the seal nor the snapshot (on an
        evaluation already sealed, the attempt may be kept as a late fact), so
        it is forgotten: sealing again seals afresh, with the output given
        then, and leaving never sends it again.
        """
        if self._snapshot is not None:
            return self._snapshot
        if not self._control_sealed:
            if self._seal_request is not None:
                await self._complete(*self._seal_request)
            else:
                output = agent_output if agent_output is not None else (self._last_reply or "")
                # Staged closure: no new call is admitted, calls already admitted may still finish, and the final
                # observations are collected after that, before the seal.
                try:
                    await self._close()
                    if self._ready:
                        await self._observe()
                finally:
                    await self._teardown()
                await self._complete(output)
        await self._read_snapshot()
        assert self._snapshot is not None
        return self._snapshot

    async def _close(self) -> None:
        with self._progress("close"):
            await self.control.close_evaluation(self.run_id, self.evaluation_id)

    async def _observe(self) -> None:
        with self._progress("observe"):
            observations = await _in_thread(self.runtime.observe)
        for source, observed in observations.items():
            await self.control.record_observations(self.run_id, self.evaluation_id, source, observed)

    async def _seal(self, agent_output: str, incomplete: tuple[str, ...] = ()) -> None:
        await self._complete(agent_output, incomplete)
        await self._read_snapshot()

    async def _complete(self, agent_output: str, incomplete: tuple[str, ...] = ()) -> None:
        self._seal_request = (agent_output, incomplete)
        marked = {"incomplete": incomplete} if incomplete else {}
        try:
            await self.control.complete_evaluation(self.run_id, self.evaluation_id, agent_output, **marked)
        except BaseException as error:
            if _refused(error):
                self._seal_request = None
            raise
        # Noted sealed before the request is forgotten: an interrupt between the two never leaves an applied seal
        # with neither marker, which cleanup would take for an episode cut short.
        self._control_sealed = True
        self._seal_request = None

    async def _read_snapshot(self) -> None:
        """Read the sealed snapshot back; on a failure, keep it as the reason, and raise it.

        The old error is cleared before the snapshot is stored, so the context
        never holds both a snapshot and a snapshot error.
        """
        environment_type = self.episode.environment_type
        assert environment_type is not None  # An episode infers one when it isn't given.
        try:
            snapshot = await self.control.snapshot(self.run_id, self.evaluation_id, environment_type=environment_type)
        except BaseException as error:
            self._snapshot_error = error
            raise
        self._snapshot_error = None
        self._snapshot = snapshot

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        failure = exc
        if failure is None:
            try:
                await self.seal()
            except BaseException as sealing:
                failure = sealing
        raised = await self._release_to_the_end(failure)
        if raised is not None and raised is not exc:
            raise raised

    @property
    def _ready(self) -> bool:
        """Whether the runtime was set up for the agent, and is still."""
        return self._setup_done and not self._torn_down

    async def _teardown(self) -> None:
        if self._set_up and not self._torn_down:
            self._torn_down = True
            with self._progress("teardown"):
                await _in_thread(self.runtime.teardown)

    async def _end_run(self) -> None:
        if self._run_started:
            self._run_started = False
            with self._progress("end_run"):
                await _in_thread(self.runtime.end_run)

    async def _release_to_the_end(self, failure: BaseException | None) -> BaseException | None:
        """Release (see :meth:`_release`) in a task of its own, to the end, and return the error to raise.

        A cancellation that arrives while the release runs doesn't stop it:
        the release is awaited to its end, however often the caller is
        cancelled. Then, of ``failure``, the cancellation and the cleanup
        steps' errors, the first that applies is raised: a KeyboardInterrupt
        or SystemExit, ``failure``, a cancellation (the caller's, then a
        step's), the first cleanup error. The others are added to it as
        notes, and ``failure`` is its context.
        """
        errors, cancelled = await _to_the_end(self._release(failure))
        ordered = [error for error in (failure, *errors) if error is not None]
        exiting = next((error for error in ordered if isinstance(error, KeyboardInterrupt | SystemExit)), None)
        step_cancelled = next((error for error in errors if isinstance(error, asyncio.CancelledError)), None)
        raised = exiting or failure or cancelled or step_cancelled or next(iter(errors), None)
        if raised is None:
            return None
        if cancelled is not None and cancelled is not raised:
            raised.add_note(f"Cancelled while releasing evaluation {self._evaluation_id}")
        for error in errors:
            if error is not raised:
                raised.add_note(f"While releasing evaluation {self._evaluation_id}: {type(error).__name__}: {error}")
        if failure is not None and failure is not raised and raised.__context__ is None:
            raised.__context__ = failure
        return raised

    async def _release(self, failure: BaseException | None) -> list[BaseException]:
        """Release what the context acquired, in reverse, and return the errors of the steps that failed.

        Every step is attempted, whatever the ones before did or raised (a
        cancellation or an interrupt included). After a failure
        or a cancellation, an evaluation that isn't sealed yet is sealed with
        what was collected, never left open: closure still lets admitted calls
        finish, a runtime that was set up is still observed, the runtime is
        torn down, and the seal marks :data:`INTERRUPTED_EVIDENCE` incomplete
        (the conversation and the agent's output were cut short, and the agent
        might have gone on to make calls it never made, so no call's absence is
        evidence). Then the session is revoked, sealed or not: nothing more is
        recorded with it, not even as a late fact. Last, the interceptor stops
        (it was live until the seal, so a call the target made after closure,
        however late, was refused and kept as a late fact) and a run the
        context started ends.

        An evaluation the control plane has sealed is never treated so, even
        when its snapshot couldn't be read back: nothing was cut short, so it
        is not interrupted, its session stays bound (a write with it is a late
        fact, as after any seal), and only the runtime, the interceptor and the
        run are released. A seal whose outcome the context didn't see (its
        response lost, or a cancellation while waiting for it; not one the
        control plane refused outright) is first sent again, as it was: the evidence was collected before it was sent, and
        if the control plane had sealed, completing again with the same output
        is a retry that leaves the snapshot as it was. Only if that fails too
        is the episode taken as interrupted. A sealed evaluation whose
        snapshot the context never tried to read (an interrupt right after
        the seal, say) has it read back here.
        """
        errors: list[BaseException] = []

        async def attempt(what: str, step: Callable[[], Awaitable[object]]) -> None:
            try:
                await step()
            except BaseException as error:
                logger.info("Couldn't %s of evaluation %s", what, self._evaluation_id, exc_info=True)
                errors.append(error)

        if failure is not None and not self._control_sealed and self._seal_request is not None:
            seal_request = self._seal_request
            await attempt("send the seal again", lambda: self._complete(*seal_request))
        if failure is not None and self._control_sealed and self._snapshot is None and self._snapshot_error is None:
            # Sealed, and its snapshot never read: the seal above, or an interrupt right after the seal's own.
            await attempt("read the snapshot back", self._read_snapshot)
        sealed = self._snapshot is not None or self._control_sealed
        if failure is not None and not sealed and self._evaluation_id is not None:
            self.interrupted = True
            await attempt("close the evaluation", self._close)
            if self._ready:
                await attempt("collect the final observations", self._observe)
            await attempt("tear down the runtime", self._teardown)
            await attempt("seal the evaluation", lambda: self._seal(self._last_reply or "", INTERRUPTED_EVIDENCE))
            await attempt("revoke the session", lambda: self.control.revoke_session(self.run_id, self.evaluation_id))
        await attempt("tear down the runtime", self._teardown)
        if self._interceptor is not None:
            await attempt("stop the interceptor", self._interceptor.stop)
        await attempt("end the runtime's run", self._end_run)
        return errors


def _refused(error: BaseException) -> bool:
    """Whether ``error`` is a refusal of a request outright: it changed nothing the request asked for.

    A ValueError is one: a rejection in process (``RejectedError``), a request
    pydantic's validation refused, or one refused before it was sent. Over
    HTTP, so is a 4xx response. A refused write can still be kept as a late
    fact (a seal with another output on a sealed evaluation, say), but the
    seal and the snapshot are as they were. Anything else (a transport error,
    a timeout, a 5xx, a cancellation) leaves the outcome unknown: the request
    may have been applied.
    """
    if isinstance(error, ValueError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and 400 <= error.response.status_code < 500


async def _to_the_end(work: Coroutine[Any, Any, T]) -> tuple[T, asyncio.CancelledError | None]:
    """Run ``work`` in a task of its own and wait for its end, however often the caller is cancelled meanwhile.

    Returns its result and the first cancellation that arrived while waiting,
    or None, for the caller to raise once the work is done.
    """
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError as error:
            if task.cancelled():  # The work itself was cancelled (the loop is shutting down): nothing to wait for.
                raise
            cancelled = cancelled or error
            if task.done():
                return task.result(), cancelled


async def _in_thread(function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a runtime step in a thread.

    A cancellation doesn't stop the thread, so the step is let finish before
    the cancellation goes on, however many arrive meanwhile: the runtime is
    never released while one of its steps is still running. Its outcome is
    then retrieved: a KeyboardInterrupt or SystemExit it raised goes on ahead
    of the cancellation (noted on it); otherwise the first cancellation goes
    on, with any other error the step raised as its context. The step is an
    executor future, not a task (as ``asyncio.to_thread`` would wrap it in):
    an interrupt it raises reaches the caller rather than leaving the event
    loop.
    """
    context = contextvars.copy_context()
    step: asyncio.Future[T] = asyncio.get_running_loop().run_in_executor(
        None, lambda: context.run(function, *args, **kwargs)
    )
    cancelled: asyncio.CancelledError | None = None
    while not step.done():
        try:
            await asyncio.wait([step])
        except asyncio.CancelledError as error:
            cancelled = cancelled or error
    if cancelled is None:
        return step.result()
    error = step.exception()
    if isinstance(error, KeyboardInterrupt | SystemExit):
        error.add_note("Cancelled while the runtime step ran")
        raise error
    if error is not None:
        cancelled.__context__ = error
    raise cancelled


def open_episode(
    episode: Episode,
    *,
    runtime: AgentRuntime | None = None,
    interception: HostInterception | None = None,
    control: ControlPlane | ControlClient | None = None,
    run_id: str | None = None,
    reset: bool = False,
    progress: Callable[[EpisodeStep], AbstractContextManager[object]] | None = None,
) -> EpisodeContext:
    """A context that runs ``episode`` once, and seals its evaluation when it's left.

    ``runtime`` is where the agent runs. By default it's a fresh one from the
    episode (``episode.new_runtime()``), whose run the context starts and
    ends; one the caller gives belongs to a run the caller has started, and
    the context only sets it up and tears it down.

    ``interception`` (a :class:`~midojo.interception_host.HostInterception`)
    runs the episode in host mode: the context starts an interceptor for its
    evaluation, given the session's recorder and the plan's tool-output part,
    and the agent gets only its tool endpoint. Without it (in-agent mode, the
    default), the agent gets the evaluation's session for an interception
    layer of its own: ``ep.control.recorder(ep.binding.session_token)`` is a
    recorder for one in this process.

    ``control`` is the control plane: by default a new one in this process,
    with no server. A :class:`~midojo.control_plane.service.ControlPlane`
    runs it in this process too. A
    :class:`~midojo.control_plane.client.ControlClient` (such as a
    ``ControlPlaneClient`` over HTTP) is used as given, and needs ``run_id``:
    the control plane then compiles the episode's suite pair from the run's
    suite, so the episode must name its ``user_task_id``. Without ``run_id``
    the episode is opened as is, outside any run.

    ``reset`` asks for the target to be reset before the episode. Reset is a
    capability the runtime declares (:class:`~midojo.runtimes.ResettableRuntime`):
    without it nothing is reset, and the result says so (``reset`` is
    ``unsupported``). The context never resets a target unasked; a caller that
    needs a clean start without the capability uses a disposable runtime.

    ``progress``, called with each :data:`EpisodeStep`, returns a context
    manager held around that step (a spinner, say).
    """
    return EpisodeContext(
        episode,
        runtime=runtime,
        interception=interception,
        control=control,
        run_id=run_id,
        reset=reset,
        progress=progress,
    )
