"""The per-process live-episode cap: N handles open together, N+1 is refused, and every failure frees its slot.

The cap counts listening interceptors and disposable sandboxes in one
process, so a driver cannot exhaust the machine by opening episodes it never
closes. These tests drive the cap through :func:`midojo.episode_handle.open`,
the public seam a driver uses, and through child processes where "per process"
is the behaviour under test. No control-plane server and no network: the
upstream is the scripted MCP server and the control plane runs in this
process.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import pytest

from midojo import episode_handle
from midojo.capabilities import MissingCapabilityError
from midojo.episode import Episode
from midojo.episode_handle import DEFAULT_MAX_LIVE_EPISODES, LiveEpisodeLimitError
from midojo.runtimes import UnmanagedRuntime
from tests.fixtures.scripted_target.upstream import scripted_upstream

ROOT = Path(__file__).resolve().parent.parent
THREADS = 8


def episode() -> Episode:
    """A one-turn unmanaged episode whose evidence needs a finished call, as the live ones do."""
    return Episode(turns=["What's the weather in Boston?"], security={"tool_called": {"name": "book_shelter"}})


def clean_env() -> dict[str, str]:
    """The test environment without the cap variable, so a process sees the documented default."""
    return {name: value for name, value in os.environ.items() if name != "MIDOJO_MAX_LIVE_EPISODES"}


def listeners() -> int:
    """How many TCP sockets this process listens on, as ``lsof`` reports them."""
    listing = subprocess.run(
        ["lsof", "-nP", "-p", str(os.getpid()), "-iTCP", "-sTCP:LISTEN", "-a"],
        capture_output=True,
        text=True,
        check=False,
    )
    return len([line for line in listing.stdout.splitlines()[1:] if line.strip()])


def port_of(handle: episode_handle.EpisodeHandle) -> int:
    endpoint = handle.tool_endpoint
    assert endpoint is not None
    return int(endpoint.url.rsplit(":", 1)[1].split("/")[0])


def hold(count: int, **options) -> list[episode_handle.EpisodeHandle]:
    """Open ``count`` handles that stay open until the test closes them."""
    return [episode_handle.open(episode(), upstream=scripted_upstream(), **options) for _ in range(count)]


def close_all(handles: list[episode_handle.EpisodeHandle]) -> None:
    for handle in handles:
        handle.close()


class FailingSetup(UnmanagedRuntime):
    """An unmanaged runtime whose setup fails: the open fails after it took a slot."""

    def setup(self, injections, **kwargs):
        raise RuntimeError("setup failed")


# --- VAL-CAP-001: N parallel handles open and close cleanly on fresh ports ---


def test_eight_threads_open_eight_handles_on_distinct_ports_twenty_rounds():
    threads_before = set(threading.enumerate())
    listeners_before = listeners()

    for round_number in range(20):
        barrier = threading.Barrier(THREADS)
        opened: list[episode_handle.EpisodeHandle | None] = [None] * THREADS
        failures: list[BaseException] = []

        def one(index: int) -> None:
            try:
                barrier.wait(timeout=30)
                opened[index] = episode_handle.open(episode(), upstream=scripted_upstream())
            except BaseException as failure:  # Every failure is collected and re-raised after the round.
                failures.append(failure)

        workers = [threading.Thread(target=one, args=(index,)) for index in range(THREADS)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)

        assert failures == [], f"round {round_number}: {failures!r}"
        assert all(worker.is_alive() is False for worker in workers)
        handles = [handle for handle in opened if handle is not None]
        assert len(handles) == THREADS
        ports = [port_of(handle) for handle in handles]
        assert len(set(ports)) == THREADS, f"round {round_number}: {ports}"
        close_all(handles)
        assert [worker for worker in workers if worker.is_alive()] == []

    assert set(threading.enumerate()) == threads_before
    assert listeners() == listeners_before


# --- VAL-CAP-002: handle N+1 is refused with LiveEpisodeLimitError, per process ---


def test_the_ninth_handle_is_refused_with_the_limit_and_binds_nothing(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EPISODES)
    try:
        assert listeners() >= DEFAULT_MAX_LIVE_EPISODES
        listeners_at_cap = listeners()
        contexts: list[str] = []
        original = episode_handle.EpisodeContext

        class Watched(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                contexts.append("constructed")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(episode_handle, "EpisodeContext", Watched)
        with pytest.raises(LiveEpisodeLimitError) as caught:
            episode_handle.open(episode(), upstream=scripted_upstream())
        error = caught.value
        assert (error.limit, error.live, error.reason) == (8, 8, "live-episode-limit")
        assert "8" in str(error) and str(DEFAULT_MAX_LIVE_EPISODES) in str(error)
        assert contexts == [], "the refusal built a context, so it ran a runtime step"
        assert listeners() == listeners_at_cap, "the refusal bound a port"
    finally:
        close_all(held)


def test_two_processes_each_keep_their_own_eight():
    script = """
import sys
from midojo import episode_handle
from midojo.episode import Episode
from tests.fixtures.scripted_target.upstream import scripted_upstream

handles = [
    episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)
]
print("open", flush=True)
sys.stdin.readline()
for handle in handles:
    handle.close()
print("closed", flush=True)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env={**clean_env(), "PYTHONPATH": str(ROOT)},
        )
        for _ in range(2)
    ]
    try:
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "open"
        # Both processes hold their eight at the same time, and neither counted the other's.
        for child in children:
            assert child.stdin is not None
            child.stdin.write("\n")
            child.stdin.flush()
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "closed"
        for child in children:
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stderr
            assert "LiveEpisodeLimitError" not in stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=30)


def test_two_threads_race_for_the_last_slot_and_exactly_one_wins(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EPISODES - 1)
    try:
        barrier = threading.Barrier(2)
        outcomes: list[episode_handle.EpisodeHandle | LiveEpisodeLimitError] = []

        def attempt() -> None:
            barrier.wait(timeout=30)
            try:
                outcomes.append(episode_handle.open(episode(), upstream=scripted_upstream()))
            except LiveEpisodeLimitError as refusal:
                outcomes.append(refusal)

        racers = [threading.Thread(target=attempt) for _ in range(2)]
        for racer in racers:
            racer.start()
        for racer in racers:
            racer.join(timeout=60)
        winners = [outcome for outcome in outcomes if isinstance(outcome, episode_handle.EpisodeHandle)]
        refusals = [outcome for outcome in outcomes if isinstance(outcome, LiveEpisodeLimitError)]
        assert len(winners) == 1 and len(refusals) == 1
        assert (refusals[0].limit, refusals[0].live) == (DEFAULT_MAX_LIVE_EPISODES, DEFAULT_MAX_LIVE_EPISODES)
        close_all(winners)
    finally:
        close_all(held)


def test_a_forked_child_owns_none_of_its_parents_slots():
    script = """
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from midojo import episode_handle
from midojo.episode import Episode
from midojo.episode_handle import LiveEpisodeLimitError
from tests.fixtures.scripted_target.upstream import scripted_upstream

held = [episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)]
print("parent open 8", flush=True)

pid = os.fork()
if pid == 0:
    try:
        mine = [episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)]
    except LiveEpisodeLimitError as inherited:
        print(f"child inherited limit={inherited.limit} live={inherited.live}", flush=True)
        sys.stdout.flush()
        os._exit(1)
    try:
        episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream())
    except LiveEpisodeLimitError as refusal:
        print(f"child refused limit={refusal.limit} live={refusal.live}", flush=True)
    else:
        print("child ninth opened", flush=True)
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

_, status = os.waitpid(pid, 0)
for handle in held:
    handle.close()
sys.exit(os.waitstatus_to_exitcode(status))
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**clean_env(), "PYTHONPATH": str(ROOT)},
        timeout=180,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == ["parent open 8", "child refused limit=8 live=8"]


# --- VAL-CAP-003: MIDOJO_MAX_LIVE_EPISODES overrides the default ---


def test_an_environment_cap_of_two_refuses_the_third(monkeypatch):
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EPISODES", "2")
    held = hold(2)
    try:
        with pytest.raises(LiveEpisodeLimitError) as caught:
            episode_handle.open(episode(), upstream=scripted_upstream())
        assert (caught.value.limit, caught.value.live) == (2, 2)
    finally:
        close_all(held)


def test_a_max_live_below_the_cap_lowers_it_and_above_it_does_not_raise_it(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    first = episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
    try:
        with pytest.raises(LiveEpisodeLimitError) as caught:
            episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
        assert caught.value.limit == 1
    finally:
        first.close()
    held = hold(DEFAULT_MAX_LIVE_EPISODES, max_live=DEFAULT_MAX_LIVE_EPISODES * 4)
    try:
        with pytest.raises(LiveEpisodeLimitError) as caught:
            episode_handle.open(episode(), upstream=scripted_upstream(), max_live=DEFAULT_MAX_LIVE_EPISODES * 4)
        assert (caught.value.limit, caught.value.live) == (DEFAULT_MAX_LIVE_EPISODES, DEFAULT_MAX_LIVE_EPISODES)
    finally:
        close_all(held)


def test_a_non_integer_environment_value_raises_before_a_slot_is_taken(monkeypatch):
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EPISODES", "many")
    with pytest.raises(ValueError, match="MIDOJO_MAX_LIVE_EPISODES"):
        episode_handle.open(episode(), upstream=scripted_upstream())
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES")
    # The invalid value took no slot: all eight are still available.
    held = hold(DEFAULT_MAX_LIVE_EPISODES)
    try:
        with pytest.raises(LiveEpisodeLimitError):
            episode_handle.open(episode(), upstream=scripted_upstream())
    finally:
        close_all(held)


def test_the_environment_cap_applies_in_a_new_process():
    script = """
from midojo import episode_handle
from midojo.episode import Episode
from midojo.episode_handle import LiveEpisodeLimitError
from tests.fixtures.scripted_target.upstream import scripted_upstream

handles = [episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream()) for _ in range(2)]
try:
    episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream())
except LiveEpisodeLimitError as error:
    print(f"limit={error.limit} live={error.live}", flush=True)
else:
    raise SystemExit("the third handle opened under a cap of two")
finally:
    for handle in handles:
        handle.close()
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**clean_env(), "PYTHONPATH": str(ROOT), "MIDOJO_MAX_LIVE_EPISODES": "2"},
        timeout=120,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "limit=2 live=2"


# --- VAL-CAP-004: slots free exactly once and failed opens consume none ---


def test_closing_one_handle_twice_frees_one_slot_only(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EPISODES)
    try:
        held[0].close()
        held[0].close()
        replacement = episode_handle.open(episode(), upstream=scripted_upstream())
        try:
            with pytest.raises(LiveEpisodeLimitError) as caught:
                episode_handle.open(episode(), upstream=scripted_upstream())
            assert (caught.value.limit, caught.value.live) == (DEFAULT_MAX_LIVE_EPISODES, DEFAULT_MAX_LIVE_EPISODES)
        finally:
            replacement.close()
    finally:
        close_all(held[1:])


def test_leaving_a_with_block_frees_the_slot_even_when_the_block_raises(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    with pytest.raises(ZeroDivisionError):
        with episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1):
            raise ZeroDivisionError("the driver broke")
    with episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1) as again:
        assert again.closed is False


def test_a_collected_handle_frees_its_slot(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)
    handle = episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
    reference = weakref.ref(handle)
    del handle
    for _ in range(100):
        gc.collect()
        if reference() is None:
            break
    assert reference() is None
    with episode_handle.open(episode(), upstream=scripted_upstream(), max_live=1):
        pass


def test_a_process_that_exits_with_an_unclosed_handle_closes_it():
    script = """
import atexit
import sys

state = {}


@atexit.register
def report():
    handle = state["handle"]
    sys.stdout.write(f"closed={handle.closed}\\n")
    sys.stdout.flush()


from midojo import episode_handle
from midojo.episode import Episode
from tests.fixtures.scripted_target.upstream import scripted_upstream

state["handle"] = episode_handle.open(Episode(turns=["hi"]), upstream=scripted_upstream())
print("open", flush=True)
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**clean_env(), "PYTHONPATH": str(ROOT)},
        timeout=120,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.splitlines() == ["open", "closed=True"]


def test_a_cap_refusal_gap_refusals_and_failed_opens_leave_every_slot_free(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EPISODES", raising=False)

    # One cap refusal.
    held = hold(DEFAULT_MAX_LIVE_EPISODES)
    try:
        with pytest.raises(LiveEpisodeLimitError) as refused:
            episode_handle.open(episode(), upstream=scripted_upstream())
        assert (refused.value.limit, refused.value.live) == (DEFAULT_MAX_LIVE_EPISODES, DEFAULT_MAX_LIVE_EPISODES)
    finally:
        close_all(held)

    # Ten runtime-gap refusals, which happen before a slot is taken.
    with pytest.MonkeyPatch.context() as gap:

        def unavailable(self, *, registry=None):
            raise MissingCapabilityError(
                gap="runtime:openshell:missing-package",
                capability="the 'openshell' runtime",
                packages=("openshell",),
            )

        gap.setattr(Episode, "check_runtime_available", unavailable)
        for _ in range(10):
            with pytest.raises(MissingCapabilityError):
                episode_handle.open(
                    Episode(runtime={"type": "openshell", "image": "pi"}, turns=["hi"]),
                    upstream=scripted_upstream(),
                )

    # Ten opens that fail after their slot is taken.
    for _ in range(10):
        with pytest.raises(RuntimeError, match="setup failed"):
            episode_handle.open(episode(), runtime=FailingSetup(), upstream=scripted_upstream())

    # Exactly eight slots are left, not one fewer for each failure.
    held = hold(DEFAULT_MAX_LIVE_EPISODES)
    try:
        with pytest.raises(LiveEpisodeLimitError) as caught:
            episode_handle.open(episode(), upstream=scripted_upstream())
        assert (caught.value.limit, caught.value.live) == (DEFAULT_MAX_LIVE_EPISODES, DEFAULT_MAX_LIVE_EPISODES)
        assert len({port_of(handle) for handle in held}) == DEFAULT_MAX_LIVE_EPISODES
    finally:
        close_all(held)
