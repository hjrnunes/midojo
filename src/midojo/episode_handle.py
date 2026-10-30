"""Drive one episode from a plain synchronous script.

:func:`open` returns an :class:`EpisodeHandle`, a synchronous wrapper over one
:class:`~midojo.episode_context.EpisodeContext`. It runs the context on a loop
thread of its own, so a script without an event loop drives a whole episode::

    from midojo.episode_handle import open

    with open(episode, upstream=klarna_sse_url) as handle:
        for message in episode.turns or ():
            handle.send_turn(message, send=my_agent)
        snapshot = handle.seal()
        verdict = handle.grade()

The handle owns the runtime's run when you give it a runtime, and the episode's
runtime when you don't: it starts the run before the evaluation opens and ends
it after the evaluation is sealed, so nothing outlives the handle. It enforces
the episode's tool policy (``Episode.tools``): the interceptor mirrors only the
allowed tools, and a blocked write is recorded as blocked and never reaches the
upstream. Closing an unsealed handle seals its evaluation as interrupted: the
conversation was cut short, so the evidence that depends on the agent going on
is incomplete.

Compatibility promise
=====================

``midojo.episode_handle`` is the synchronous driver API. The names in
:data:`HANDLE_METHODS`, the exception classes in this module and their
``reason`` strings, :data:`HANDLE_API_VERSION` and
:data:`DEFAULT_MAX_LIVE_EPISODES` are stable within a 0.x minor release. New
keyword arguments to :func:`open` and :meth:`EpisodeHandle.send_turn` are
appended, with defaults, so existing calls keep working. A change that breaks
any of these raises the major version.

The asynchronous API is unchanged: :func:`midojo.open_episode` and
:class:`~midojo.episode_context.EpisodeContext` keep their behavior, and a
handle is exactly a context driven on a thread.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import dataclasses
import logging
import os
import subprocess
import sys
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Literal

from midojo.control_plane.grading import Verdict
from midojo.control_plane.grading import grade as grade_snapshot
from midojo.control_plane.state import Snapshot, Turn
from midojo.episode import Episode, InterceptionSettings, ToolPolicy
from midojo.episode_context import EpisodeContext, EpisodeResult, EpisodeStep
from midojo.episode_document import EpisodeDocumentError
from midojo.interception_host import HostInterception
from midojo.runtimes import AgentRuntime, RuntimeCapabilities, is_disposable, runtime_capabilities
from midojo.tool_endpoint import ToolEndpoint

logger = logging.getLogger(__name__)

HANDLE_API_VERSION = 1
"""The synchronous handle's API version: see the module's compatibility promise."""

DEFAULT_MAX_LIVE_EPISODES = 8
"""How many handles one process keeps open at once, unless ``MIDOJO_MAX_LIVE_EPISODES`` says otherwise."""

MAX_LIVE_ENV = "MIDOJO_MAX_LIVE_EPISODES"
"""The environment variable that overrides :data:`DEFAULT_MAX_LIVE_EPISODES`."""

HANDLE_METHODS = ("tool_block", "start_turn", "record_reply", "send_turn", "seal")
"""The handle's driving methods: once the handle is closed, each raises :class:`HandleClosedError`.

``close`` is not one of them: closing twice is a no-op. ``grade`` is not
either: it is a pure read of the sealed snapshot, so it still works after a
close, which is how a script grades an evaluation its close sealed as
interrupted.
"""

# The loop thread's name: one per handle, and nothing else in a driver script uses it.
_LOOP_THREAD_NAME = "midojo-episode-handle"

# How long a close waits for the loop thread to stop, and for a reaper to end.
_STOP_SECONDS = 30.0
_REAPER_STOP_SECONDS = 5.0


class HandleError(RuntimeError):
    """A synchronous handle refused a call. Every subclass names itself in ``reason``."""

    reason = "handle-error"


class HandleOwnerError(HandleError):
    """The handle was used from another process: its loop thread and its context live in the owning one."""

    reason = "handle-owner"


class HandleClosedError(HandleError):
    """The handle is closed: its evaluation is sealed and its materials are released."""

    reason = "handle-closed"


class SendTurnUnsupported(HandleError):  # noqa: N818 - the handle contract names it, not a lint choice.
    """The runtime doesn't run the agent itself, so the handle can't send a turn to it without ``send``."""

    reason = "send-turn-unsupported"


class RepeatWithoutResetError(HandleError):
    """A run key repeats on a runtime that neither resets nor replaces its target, so the attempts mix."""

    reason = "repeat-without-reset"


class LiveEpisodeLimitError(HandleError):
    """Too many handles are open in this process; one must close before another opens.

    ``limit`` is the limit in force, ``live`` how many handles held a slot.
    """

    reason = "live-episode-limit"

    def __init__(self, limit: int, live: int) -> None:
        self.limit = limit
        self.live = live
        super().__init__(
            f"{live} live episodes is the process's limit of {limit}; close a handle before opening another, or "
            f"raise {MAX_LIVE_ENV}"
        )


class _ClosedByCaller(Exception):  # noqa: N818 - it marks a close, and never reaches a caller as an error.
    """What a handle tells its context when the caller closes it before the seal: the episode was cut short."""


# --- the live-episode cap, one count per process ---

_live_lock = threading.Lock()
_live = 0
_live_tokens: set[str] = set()


def _limit(max_live: int | None) -> int:
    """The cap in force: the environment's or the default's, lowered by ``max_live`` and never raised by it."""
    raw = os.environ.get(MAX_LIVE_ENV)
    limit = DEFAULT_MAX_LIVE_EPISODES
    if raw is not None:
        try:
            limit = int(raw)
        except ValueError:
            limit = 0
        if limit < 1:
            raise ValueError(f"{MAX_LIVE_ENV} must be a positive integer, not {raw!r}")
    if max_live is not None:
        if isinstance(max_live, bool) or not isinstance(max_live, int):
            raise TypeError(f"max_live must be an int, not {type(max_live).__name__}")
        if max_live < 1:
            raise ValueError(f"max_live must be a positive integer, not {max_live}")
        limit = min(limit, max_live)
    return limit


def _acquire_slot(limit: int) -> str:
    """Take a live-episode slot, or raise :class:`LiveEpisodeLimitError` without binding anything."""
    global _live
    with _live_lock:
        if _live >= limit:
            raise LiveEpisodeLimitError(limit, _live)
        _live += 1
        token = uuid.uuid4().hex
        _live_tokens.add(token)
    return token


def _release_slot(token: str) -> None:
    """Give a slot back, once: a token that was already released is not counted again."""
    global _live
    with _live_lock:
        if token not in _live_tokens:
            return
        _live_tokens.discard(token)
        _live = max(0, _live - 1)


# --- attempted run keys, one registry per process ---

_attempts: dict[str, int] = {}


def _check_repeat(repeat_of: str | None, capabilities: RuntimeCapabilities, reset: bool) -> None:
    """Refuse a repeated run key on a runtime that leaves the previous attempt in place."""
    if repeat_of is None:
        return
    if type(repeat_of) is not str or not repeat_of:
        raise ValueError("repeat_of must be a non-empty str or None")
    attempts = _attempts.get(repeat_of, 0)
    _attempts[repeat_of] = attempts + 1
    isolated = capabilities.isolates_attempts or (reset and capabilities.resettable)
    if attempts and not isolated:
        raise RepeatWithoutResetError(
            f"run key {repeat_of!r} was already attempted in this process on the {capabilities.name!r} runtime, "
            "which neither resets nor replaces its target between attempts; pass reset=True or a disposable runtime"
        )


# --- the loop thread ---


class _Loop:
    """One event loop on one daemon thread, driven synchronously from any thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, name=_LOOP_THREAD_NAME, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self._drain()

    def _drain(self) -> None:
        """Cancel what the loop still runs and stop its threads, so it closes without a leak."""
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())

    def call(self, work: Any) -> Any:
        """Run ``work`` on the loop and return its result, waiting for it; its error is raised here."""
        if not self.thread.is_alive():
            work.close()
            raise HandleError("The handle's loop thread is gone: the handle is closed")
        return asyncio.run_coroutine_threadsafe(work, self.loop).result()

    def stop(self) -> None:
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(_STOP_SECONDS)
        if not self.thread.is_alive():
            self.loop.close()


async def _in_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a runtime step in a thread, so the handle's loop is never blocked by one."""
    return await asyncio.to_thread(function, *args, **kwargs)


# --- what a handle owns until it closes ---


class _Resource:
    """The loop, the context and the live-episode slot one handle owns, freed exactly once.

    It holds no reference to the handle, so the handle's garbage collection
    or the interpreter's exit can close it through a finalizer.
    """

    def __init__(
        self,
        *,
        episode: Episode,
        context: EpisodeContext,
        loop: _Loop,
        token: str,
        runtime: AgentRuntime,
        ends_run: bool,
        reaper: subprocess.Popen[bytes] | None,
    ) -> None:
        self.episode = episode
        self.context = context
        self.loop = loop
        self.token = token
        self.runtime = runtime
        self.ends_run = ends_run
        self.reaper = reaper
        self.lock = threading.RLock()
        self.owner_pid = os.getpid()
        self.evaluation_id = context.evaluation_id
        self.binding = context.binding
        self.tool_endpoint = self.binding.tool_endpoint
        self.capabilities = runtime_capabilities(runtime)
        self.turns: list[Turn] = []
        self.closed = False

    # --- what the handle reads ---

    @property
    def sealed(self) -> bool:
        try:
            self.context.result
        except RuntimeError:
            return False
        return True

    @property
    def result(self) -> EpisodeResult | None:
        """The run's result once its evaluation is sealed, and None before: the async property stays strict."""
        try:
            return self.context.result
        except RuntimeError:
            return None

    # --- calls ---

    def usable(self) -> None:
        """Refuse a call from another process, or on a closed handle, before it touches anything."""
        if os.getpid() != self.owner_pid:
            raise HandleOwnerError(
                f"Episode handle {self.evaluation_id} belongs to process {self.owner_pid}; this is process "
                f"{os.getpid()}. Open a handle in this process instead."
            )
        if self.closed:
            raise HandleClosedError(f"Episode handle {self.evaluation_id} is closed")

    # --- release ---

    def shutdown(self, failure: BaseException | None = None) -> None:
        """Close the context once, sealing an unsealed evaluation as interrupted, then release everything."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            marker = (
                failure
                if failure is not None
                else _ClosedByCaller(f"handle {self.evaluation_id} closed before its seal")
            )
            if self.loop.thread.is_alive():
                try:
                    self.loop.call(self.context.__aexit__(type(marker), marker, marker.__traceback__))
                except BaseException as raised:
                    if raised is not marker:
                        logger.info("Couldn't release evaluation %s", self.evaluation_id, exc_info=True)
                if self.ends_run:
                    try:
                        self.loop.call(_in_thread(self.runtime.end_run))
                    except BaseException:
                        logger.info("Couldn't end the run of evaluation %s", self.evaluation_id, exc_info=True)
            self._stop_reaper()
            self.loop.stop()
            _release_slot(self.token)

    def _stop_reaper(self) -> None:
        process = self.reaper
        self.reaper = None
        _stop_reaper(process)


_sessions: weakref.WeakSet[_Resource] = weakref.WeakSet()


def _abandon(resource: _Resource) -> None:
    """Close a resource whose handle was garbage collected or whose process is exiting."""
    try:
        resource.shutdown()
    except BaseException:
        logger.info("Couldn't close an abandoned episode handle", exc_info=True)


@atexit.register
def _close_open_handles() -> None:
    for resource in list(_sessions):
        _abandon(resource)


def _reset_after_fork() -> None:
    """Start a forked child's live-episode count from zero: it owns none of the parent's episodes.

    Threads don't survive a fork, so the parent's interceptors aren't serving
    in the child and its handles are unusable there (:class:`HandleOwnerError`).
    Counting them would make the child refuse episodes it has room for, so the
    child gets a fresh lock, an empty count and an empty session set.
    """
    global _live, _live_lock
    _live_lock = threading.Lock()
    _live = 0
    _live_tokens.clear()
    _sessions.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


# --- the sandbox reaper: a driver killed mid-run leaves no sandbox ---


def _reaper(runtime: AgentRuntime) -> subprocess.Popen[bytes] | None:
    """A detached process that removes this handle's workspace once the driver is gone.

    A sandbox lives in its own run's workspace, so removing that workspace's
    sandboxes and then the workspace frees everything this handle made,
    wherever the driver dies. The reaper reads the gateway the runtime used and
    is otherwise independent of it.
    """
    workspace = getattr(runtime, "workspace_name", "")
    if not is_disposable(runtime) or not workspace:
        return None
    cluster = getattr(runtime, "cluster", "") or ""
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "midojo._sandbox_reaper",
                "--workspace",
                workspace,
                "--driver",
                str(os.getpid()),
                "--cluster",
                cluster,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        logger.info("Couldn't start the sandbox reaper for %s", workspace, exc_info=True)
        return None


def _stop_reaper(process: subprocess.Popen[bytes] | None) -> None:
    """Stop a reaper before it acts: the runtime's own teardown, or a failed open, freed the workspace."""
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_REAPER_STOP_SECONDS)
    except BaseException:
        logger.info("Couldn't stop the sandbox reaper", exc_info=True)
        if process.poll() is None:
            process.kill()


def _reaper_after_start_run(start: Callable[[], None]) -> Callable[[EpisodeStep], AbstractContextManager[object]]:
    """The context's progress steps, plus a ``start`` as soon as its own ``start_run`` made the workspace.

    A step's context manager leaves once the step returned or raised, so the
    reaper watches a run the context started before ``setup`` creates its
    sandbox: a driver killed during setup still leaves nothing behind.
    """

    @contextlib.contextmanager
    def progress(step: EpisodeStep) -> Iterator[None]:
        try:
            yield
        finally:
            if step == "start_run":
                start()

    return progress


# --- the handle ---


class EpisodeHandle:
    """One episode, driven synchronously: see :func:`open` and the module's compatibility promise.

    ``evaluation_id``, ``binding``, ``tool_endpoint`` and ``capabilities``
    name the open evaluation and what to give the agent. ``turns`` are the
    turns it drove, in order; ``sealed``, ``closed`` and ``result`` stay
    readable after a close, and every method in :data:`HANDLE_METHODS` raises
    :class:`HandleClosedError` once it is closed.
    """

    def __init__(self, resource: _Resource) -> None:
        self._resource = resource

    # --- what the host reads ---

    @property
    def evaluation_id(self) -> str:
        return self._resource.evaluation_id

    @property
    def binding(self) -> Any:
        """What to give the agent: its tool endpoint (host mode) or the evaluation's session."""
        return self._resource.binding

    @property
    def tool_endpoint(self) -> ToolEndpoint | None:
        """The interceptor's endpoint, or None when the agent talks to the control plane instead."""
        return self._resource.tool_endpoint

    @property
    def capabilities(self) -> RuntimeCapabilities:
        """What the episode's runtime can do, answered without building another."""
        return self._resource.capabilities

    @property
    def turns(self) -> tuple[Turn, ...]:
        """The turns this handle drove, in order."""
        return tuple(self._resource.turns)

    @property
    def sealed(self) -> bool:
        """Whether the evaluation is sealed: by :meth:`seal`, or as interrupted by :meth:`close`."""
        return self._resource.sealed

    @property
    def closed(self) -> bool:
        """Whether the handle released its context, loop thread and live-episode slot."""
        return self._resource.closed

    @property
    def result(self) -> EpisodeResult | None:
        """The run's result once sealed, and None before."""
        return self._resource.result

    # --- the tool block ---

    def tool_block(self, label: str = "midojo") -> dict[str, Any]:
        """The interceptor as an MCP tool block, for a target that takes one.

        ``server_url`` is the endpoint and ``authorization`` its bearer token;
        ``allowed_tools`` lists what the episode's policy exposes, so the
        target is offered exactly the tools the interceptor serves.
        """
        with self._resource.lock:
            self._resource.usable()
            endpoint = self._resource.tool_endpoint
            if endpoint is None:
                raise HandleError(
                    f"Episode {self.evaluation_id} runs without host interception, so it has no tool endpoint; "
                    "open it with an upstream for a tool block"
                )
            interceptor = self._resource.context.interceptor
            names = self._resource.loop.call(interceptor.tool_names()) if interceptor is not None else []
        return {
            "type": "mcp",
            "server_label": label,
            "server_url": endpoint.url,
            "authorization": endpoint.token,
            "require_approval": "never",
            "allowed_tools": list(names),
        }

    # --- turns ---

    def start_turn(self, message: str) -> int:
        """Start the next turn with the user's ``message``; calls the agent makes from now on carry its index."""
        with self._resource.lock:
            self._resource.usable()
            return self._start(message)

    def _start(self, message: str) -> int:
        if type(message) is not str:
            raise TypeError(f"a turn's message must be a str, not {type(message).__name__}")
        index = self._resource.loop.call(self._resource.context.start_turn(message))
        self._resource.turns.append(Turn(index=index, message=message))
        return index

    def record_reply(self, index: int, reply: str) -> None:
        """Record the agent's ``reply`` to turn ``index``; the last reply is the agent's output at the seal."""
        with self._resource.lock:
            self._resource.usable()
            self._record(index, reply)

    def _record(self, index: int, reply: str) -> None:
        self._resource.loop.call(self._resource.context.record_turn_reply(index, reply))
        turns = self._resource.turns
        for position, turn in enumerate(turns):
            if turn.index == index:
                turns[position] = dataclasses.replace(turn, reply=reply)
                return

    def send_turn(self, message: str, *, send: Any = None, timeout: float | None = None) -> str:
        """Start a turn, send ``message`` to the agent, record its reply, and return the reply.

        ``send`` is a callable that takes the message and returns the agent's
        reply, for an agent the caller drives. Without it, the runtime runs the
        agent itself (as OpenShell does): the handle runs its ``exec_agent``
        command in the sandbox and returns the agent's output. On a runtime
        that does neither, :class:`SendTurnUnsupported` is raised at once.
        """
        with self._resource.lock:
            self._resource.usable()
            if send is None and self._runtime_send is None:
                raise SendTurnUnsupported(
                    f"the {self._resource.capabilities.name!r} runtime doesn't run the agent, so handle "
                    f"{self.evaluation_id} needs send= to send a turn"
                )
            index = self._start(message)
            if send is not None:
                reply = send(message)
            else:
                reply = self._send_with_runtime(message, index, timeout)
            if type(reply) is not str:
                raise TypeError(f"an agent's reply must be a str, not {type(reply).__name__}")
            self._record(index, reply)
            return reply

    @property
    def _runtime_send(self) -> Any:
        """The runtime's agent-running step, when it runs the agent itself."""
        exec_agent = getattr(self._resource.runtime, "exec_agent", None)
        return exec_agent if callable(exec_agent) else None

    def _send_with_runtime(self, message: str, index: int, timeout: float | None) -> str:
        runtime = self._resource.runtime
        count = len(self._resource.episode.turns or ()) or index + 1
        seconds = timeout if timeout is not None else getattr(runtime, "agent_timeout_seconds", 600.0)
        result = self._resource.loop.call(
            _in_thread(
                self._runtime_send,
                message,
                timeout_seconds=seconds,
                env={"MIDOJO_TURN_INDEX": str(index), "MIDOJO_TURN_COUNT": str(count)},
            )
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"OpenShell agent exited with code {result.exit_code} on turn {index + 1}/{count}: {result.stderr}"
            )
        return str(result.stdout).strip()

    # --- sealing and grading ---

    def seal(self, agent_output: str | None = None) -> Snapshot:
        """Close the evaluation, collect the runtime's observations, tear it down and seal it.

        ``agent_output`` defaults to the last recorded reply. Sealing before
        closing keeps the interceptor up while the host grades the snapshot, so
        a call after the seal is a late fact, not a lost one.
        """
        with self._resource.lock:
            self._resource.usable()
            if self._resource.sealed:
                return self._resource.context.sealed
            return self._resource.loop.call(self._resource.context.seal(agent_output))

    def grade(self, judgements: Sequence[Any] | None = None) -> Verdict:
        """Grade the sealed snapshot with the episode's checks, naming what was graded.

        This is a pure function of the episode and the sealed snapshot, so it
        works after a close too: a close that sealed an unsealed evaluation as
        interrupted leaves a grade of that interrupted evidence.
        """
        with self._resource.lock:
            if os.getpid() != self._resource.owner_pid:
                self._resource.usable()  # Raises HandleOwnerError.
            snapshot = self._resource.context.sealed
            return grade_snapshot(self._resource.episode, snapshot, judgements=judgements)

    # --- closing ---

    def close(self) -> None:
        """Seal an unsealed evaluation as interrupted, then release the context, thread and slot.

        Closing twice is a no-op. A call from another process is refused and
        changes nothing.
        """
        resource = self._resource
        if os.getpid() != resource.owner_pid:
            resource.usable()  # Raises HandleOwnerError, before anything touches the inherited state.
        resource.shutdown()

    def __enter__(self) -> EpisodeHandle:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Literal[False]:
        resource = self._resource
        with resource.lock:
            if exc is None:
                try:
                    if not resource.sealed:
                        resource.loop.call(resource.context.seal())
                except BaseException as sealing:
                    # A seal that failed still releases what the handle holds, and the error goes to the caller.
                    resource.shutdown(sealing)
                    raise
            resource.shutdown(exc)
        return False


# --- opening ---


def _resolve_upstream(episode: Episode, upstream: Any, interception: HostInterception | None) -> Any:
    """The MCP server the interceptor mirrors: the explicit interception's, the argument's, the episode's."""
    if interception is not None:
        return interception.upstream
    if upstream is not None:
        return upstream
    if episode.interception is not None and episode.interception.upstream is not None:
        return episode.interception.upstream
    raise EpisodeDocumentError(
        "upstream not supplied: pass upstream= to open, or set interception.upstream in the episode's document"
    )


def _tool_policy(tools: ToolPolicy | None) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """The episode's tool policy as the interceptor's allowlist and block list."""
    if tools is None:
        return None, ()
    names = {*tools.allowed, *tools.write_disposition}
    default = tools.write_disposition_default
    blocked = tuple(sorted(name for name in names if tools.write_disposition.get(name, default) == "block"))
    return frozenset(tools.allowed), blocked


def _interception(episode: Episode, upstream: Any, given: HostInterception | None) -> HostInterception:
    """The host interception one evaluation runs: the given one, or the episode's settings, with its tool policy."""
    settings = given
    if settings is None and episode.interception is not None:
        configured: InterceptionSettings = episode.interception
        options: dict[str, Any] = {"upstream": upstream, "observe_requests": configured.observe_requests}
        if configured.bind_host is not None:
            options["bind_host"] = configured.bind_host
        if configured.port is not None:
            options["port"] = configured.port
        if configured.advertise_host is not None:
            options["advertise_host"] = configured.advertise_host
        settings = HostInterception(**options)
    if settings is None:
        settings = HostInterception(upstream=upstream)
    allowed, blocked = _tool_policy(episode.tools)
    return dataclasses.replace(settings, allowed=allowed, blocked=blocked)


def _run_is_open(runtime: AgentRuntime) -> bool:
    """Whether the runtime's run is already open, which means the caller started it."""
    workspace = getattr(runtime, "workspace_name", None)
    return isinstance(workspace, str) and bool(workspace)


def open(
    episode: Episode,
    *,
    runtime: AgentRuntime | None = None,
    upstream: Any = None,
    interception: HostInterception | None = None,
    control: Any = None,
    run_id: str | None = None,
    reset: bool = False,
    max_live: int | None = None,
    repeat_of: str | None = None,
) -> EpisodeHandle:
    """Open ``episode`` as a synchronous handle: one context, one loop thread, one live-episode slot.

    ``runtime`` is the agent's runtime; without one, the episode's own runtime
    is built, which is why the episode's capabilities are checked first
    (:meth:`~midojo.episode.Episode.check_runtime_available`). ``upstream`` is
    the MCP server the interceptor mirrors; without it, the episode's
    ``interception.upstream``, and without either the call raises
    :class:`~midojo.episode_document.EpisodeDocumentError` before anything is
    acquired. ``interception`` is a whole host-interception setting that wins
    over the episode's, which is how a caller changes the port or the bind
    host; the episode's tool policy still applies. ``control`` is a control
    plane or client; without one, the episode runs in a control plane in this
    process. ``max_live`` lowers this process's cap (never raises it), and
    ``repeat_of`` is the key of the run this episode repeats, which a runtime
    that isolates neither by reset nor by a fresh target refuses.
    """
    if not isinstance(episode, Episode):
        raise TypeError(f"open expects an Episode, not {type(episode).__name__}")
    upstream = _resolve_upstream(episode, upstream, interception)
    if runtime is None:
        # Before a context, a runtime or a port exists: an uninstalled provider is a named gap.
        episode.check_runtime_available()
    capabilities = runtime_capabilities(episode.runtime if runtime is None else runtime)
    _check_repeat(repeat_of, capabilities, reset)
    token = _acquire_slot(_limit(max_live))
    loop: _Loop | None = None
    context: EpisodeContext | None = None
    started_run = False
    reaper: subprocess.Popen[bytes] | None = None

    def start_reaper() -> None:
        """Watch the run's workspace, wherever the run was started: the handle, or the context it gave none."""
        nonlocal reaper
        if reaper is None and context is not None:
            reaper = _reaper(context.runtime)

    try:
        context = EpisodeContext(
            episode,
            runtime=runtime,
            interception=_interception(episode, upstream, interception),
            control=control,
            run_id=run_id,
            reset=reset,
            # A context that builds the runtime starts its run itself: watch it as soon as start_run returns.
            progress=_reaper_after_start_run(start_reaper),
        )
        loop = _Loop()
        if runtime is not None and hasattr(runtime, "start_run") and not _run_is_open(runtime):
            loop.call(
                _in_thread(
                    runtime.start_run,
                    run_id or f"episode-{uuid.uuid4().hex[:12]}",
                    suite_name=episode.suite_name or "episode",
                )
            )
            started_run = True
            # The run's workspace exists now, before setup makes the sandbox: a driver killed from here on is reaped.
            start_reaper()
        loop.call(context.__aenter__())
    except BaseException:
        if loop is not None:
            if context is not None and started_run and runtime is not None and loop.thread.is_alive():
                try:
                    loop.call(_in_thread(runtime.end_run))
                except BaseException:
                    logger.info("Couldn't end a run whose evaluation never opened", exc_info=True)
            loop.stop()
        _stop_reaper(reaper)
        _release_slot(token)
        raise
    assert loop is not None and context is not None
    built = context.runtime
    resource = _Resource(
        episode=episode,
        context=context,
        loop=loop,
        token=token,
        runtime=built,
        ends_run=started_run,
        reaper=reaper,
    )
    handle = EpisodeHandle(resource)
    _sessions.add(resource)
    finalizer = weakref.finalize(handle, _abandon, resource)
    finalizer.atexit = False
    return handle
