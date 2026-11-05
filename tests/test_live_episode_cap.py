"""The per-process live-episode cap: N handles open together, N+1 is refused, and every failure frees its slot.

The cap counts listening interceptors and disposable sandboxes in one
process, so a driver cannot exhaust the machine by opening episodes it never
closes. These tests drive the cap through :func:`midojo.evaluation_handle.open`,
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

from midojo import evaluation_handle
from midojo.capabilities import MissingCapabilityError
from midojo.evaluation_handle import DEFAULT_MAX_LIVE_EVALUATIONS, LiveEvaluationLimitError
from midojo.injection_spec import InjectionSpec
from midojo.runtimes import UnmanagedRuntime
from tests.fixtures.scripted_target.upstream import scripted_upstream

ROOT = Path(__file__).resolve().parent.parent
THREADS = 8


def episode() -> InjectionSpec:
    """A one-turn unmanaged episode whose evidence needs a finished call, as the live ones do."""
    return InjectionSpec(turns=["What's the weather in Boston?"], security={"tool_called": {"name": "book_shelter"}})


def clean_env() -> dict[str, str]:
    """The test environment without the cap variable, so a process sees the documented default."""
    return {name: value for name, value in os.environ.items() if name != "MIDOJO_MAX_LIVE_EVALUATIONS"}


def listeners() -> int:
    """How many TCP sockets this process listens on, as ``lsof`` reports them."""
    listing = subprocess.run(
        ["lsof", "-nP", "-p", str(os.getpid()), "-iTCP", "-sTCP:LISTEN", "-a"],
        capture_output=True,
        text=True,
        check=False,
    )
    return len([line for line in listing.stdout.splitlines()[1:] if line.strip()])


def port_of(handle: evaluation_handle.EvaluationHandle) -> int:
    endpoint = handle.tool_endpoint
    assert endpoint is not None
    return int(endpoint.url.rsplit(":", 1)[1].split("/")[0])


def hold(count: int, **options) -> list[evaluation_handle.EvaluationHandle]:
    """Open ``count`` handles that stay open until the test closes them."""
    return [evaluation_handle.open(episode(), upstream=scripted_upstream(), **options) for _ in range(count)]


def close_all(handles: list[evaluation_handle.EvaluationHandle]) -> None:
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
        opened: list[evaluation_handle.EvaluationHandle | None] = [None] * THREADS
        failures: list[BaseException] = []

        def one(index: int) -> None:
            try:
                barrier.wait(timeout=30)
                opened[index] = evaluation_handle.open(episode(), upstream=scripted_upstream())
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


# --- VAL-CAP-002: handle N+1 is refused with LiveEvaluationLimitError, per process ---


def test_the_ninth_handle_is_refused_with_the_limit_and_binds_nothing(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS)
    try:
        assert listeners() >= DEFAULT_MAX_LIVE_EVALUATIONS
        listeners_at_cap = listeners()
        contexts: list[str] = []
        original = evaluation_handle.EvaluationContext

        class Watched(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                contexts.append("constructed")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(evaluation_handle, "EvaluationContext", Watched)
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream())
        error = caught.value
        assert (error.limit, error.live, error.reason) == (8, 8, "live-evaluation-limit")
        assert "8" in str(error) and str(DEFAULT_MAX_LIVE_EVALUATIONS) in str(error)
        assert contexts == [], "the refusal built a context, so it ran a runtime step"
        assert listeners() == listeners_at_cap, "the refusal bound a port"
    finally:
        close_all(held)


def test_two_processes_each_keep_their_own_eight():
    script = """
import sys
from midojo import evaluation_handle
from midojo.injection_spec import InjectionSpec
from tests.fixtures.scripted_target.upstream import scripted_upstream

handles = [
    evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)
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
            assert "LiveEvaluationLimitError" not in stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=30)


def test_two_threads_race_for_the_last_slot_and_exactly_one_wins(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS - 1)
    try:
        barrier = threading.Barrier(2)
        outcomes: list[evaluation_handle.EvaluationHandle | LiveEvaluationLimitError] = []

        def attempt() -> None:
            barrier.wait(timeout=30)
            try:
                outcomes.append(evaluation_handle.open(episode(), upstream=scripted_upstream()))
            except LiveEvaluationLimitError as refusal:
                outcomes.append(refusal)

        racers = [threading.Thread(target=attempt) for _ in range(2)]
        for racer in racers:
            racer.start()
        for racer in racers:
            racer.join(timeout=60)
        winners = [outcome for outcome in outcomes if isinstance(outcome, evaluation_handle.EvaluationHandle)]
        refusals = [outcome for outcome in outcomes if isinstance(outcome, LiveEvaluationLimitError)]
        assert len(winners) == 1 and len(refusals) == 1
        assert (refusals[0].limit, refusals[0].live) == (DEFAULT_MAX_LIVE_EVALUATIONS, DEFAULT_MAX_LIVE_EVALUATIONS)
        close_all(winners)
    finally:
        close_all(held)


def test_a_forked_child_owns_none_of_its_parents_slots():
    script = """
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from midojo import evaluation_handle
from midojo.injection_spec import InjectionSpec
from midojo.evaluation_handle import LiveEvaluationLimitError
from tests.fixtures.scripted_target.upstream import scripted_upstream

held = [evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)]
print("parent open 8", flush=True)

pid = os.fork()
if pid == 0:
    try:
        mine = [evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream()) for _ in range(8)]
    except LiveEvaluationLimitError as inherited:
        print(f"child inherited limit={inherited.limit} live={inherited.live}", flush=True)
        sys.stdout.flush()
        os._exit(1)
    try:
        evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream())
    except LiveEvaluationLimitError as refusal:
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


# --- VAL-CAP-003: MIDOJO_MAX_LIVE_EVALUATIONS overrides the default ---


def test_an_environment_cap_of_two_refuses_the_third(monkeypatch):
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EVALUATIONS", "2")
    held = hold(2)
    try:
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream())
        assert (caught.value.limit, caught.value.live) == (2, 2)
    finally:
        close_all(held)


def test_the_old_environment_name_still_sets_the_cap_with_a_deprecation_warning(monkeypatch):
    """The variable was MIDOJO_MAX_LIVE_EPISODES: a host that still sets it keeps its cap, with a warning."""
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EPISODES", "1")
    with pytest.warns(DeprecationWarning, match="MIDOJO_MAX_LIVE_EVALUATIONS"):
        held = hold(1)
        try:
            with pytest.raises(LiveEvaluationLimitError) as caught:
                evaluation_handle.open(episode(), upstream=scripted_upstream())
            assert (caught.value.limit, caught.value.live) == (1, 1)
        finally:
            close_all(held)


def test_the_new_environment_name_wins_over_the_old_one(monkeypatch):
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EPISODES", "1")
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EVALUATIONS", "2")
    held = hold(2)
    try:
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream())
        assert caught.value.limit == 2
    finally:
        close_all(held)


def test_a_max_live_below_the_cap_lowers_it_and_above_it_does_not_raise_it(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    first = evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
    try:
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
        assert caught.value.limit == 1
    finally:
        first.close()
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS, max_live=DEFAULT_MAX_LIVE_EVALUATIONS * 4)
    try:
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=DEFAULT_MAX_LIVE_EVALUATIONS * 4)
        assert (caught.value.limit, caught.value.live) == (DEFAULT_MAX_LIVE_EVALUATIONS, DEFAULT_MAX_LIVE_EVALUATIONS)
    finally:
        close_all(held)


def test_a_non_integer_environment_value_raises_before_a_slot_is_taken(monkeypatch):
    monkeypatch.setenv("MIDOJO_MAX_LIVE_EVALUATIONS", "many")
    with pytest.raises(ValueError, match="MIDOJO_MAX_LIVE_EVALUATIONS"):
        evaluation_handle.open(episode(), upstream=scripted_upstream())
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS")
    # The invalid value took no slot: all eight are still available.
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS)
    try:
        with pytest.raises(LiveEvaluationLimitError):
            evaluation_handle.open(episode(), upstream=scripted_upstream())
    finally:
        close_all(held)


def test_the_environment_cap_applies_in_a_new_process():
    script = """
from midojo import evaluation_handle
from midojo.injection_spec import InjectionSpec
from midojo.evaluation_handle import LiveEvaluationLimitError
from tests.fixtures.scripted_target.upstream import scripted_upstream

handles = [evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream()) for _ in range(2)]
try:
    evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream())
except LiveEvaluationLimitError as error:
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
        env={**clean_env(), "PYTHONPATH": str(ROOT), "MIDOJO_MAX_LIVE_EVALUATIONS": "2"},
        timeout=120,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "limit=2 live=2"


# --- VAL-CAP-004: slots free exactly once and failed opens consume none ---


def test_closing_one_handle_twice_frees_one_slot_only(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS)
    try:
        held[0].close()
        held[0].close()
        replacement = evaluation_handle.open(episode(), upstream=scripted_upstream())
        try:
            with pytest.raises(LiveEvaluationLimitError) as caught:
                evaluation_handle.open(episode(), upstream=scripted_upstream())
            assert (caught.value.limit, caught.value.live) == (
                DEFAULT_MAX_LIVE_EVALUATIONS,
                DEFAULT_MAX_LIVE_EVALUATIONS,
            )
        finally:
            replacement.close()
    finally:
        close_all(held[1:])


def test_leaving_a_with_block_frees_the_slot_even_when_the_block_raises(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    with pytest.raises(ZeroDivisionError):
        with evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1):
            raise ZeroDivisionError("the driver broke")
    with evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1) as again:
        assert again.closed is False


def test_a_collected_handle_frees_its_slot(monkeypatch):
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)
    handle = evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1)
    reference = weakref.ref(handle)
    del handle
    for _ in range(100):
        gc.collect()
        if reference() is None:
            break
    assert reference() is None
    with evaluation_handle.open(episode(), upstream=scripted_upstream(), max_live=1):
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


from midojo import evaluation_handle
from midojo.injection_spec import InjectionSpec
from tests.fixtures.scripted_target.upstream import scripted_upstream

state["handle"] = evaluation_handle.open(InjectionSpec(turns=["hi"]), upstream=scripted_upstream())
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
    monkeypatch.delenv("MIDOJO_MAX_LIVE_EVALUATIONS", raising=False)

    # One cap refusal.
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS)
    try:
        with pytest.raises(LiveEvaluationLimitError) as refused:
            evaluation_handle.open(episode(), upstream=scripted_upstream())
        assert (refused.value.limit, refused.value.live) == (DEFAULT_MAX_LIVE_EVALUATIONS, DEFAULT_MAX_LIVE_EVALUATIONS)
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

        gap.setattr(InjectionSpec, "check_runtime_available", unavailable)
        for _ in range(10):
            with pytest.raises(MissingCapabilityError):
                evaluation_handle.open(
                    InjectionSpec(runtime={"type": "openshell", "image": "pi"}, turns=["hi"]),
                    upstream=scripted_upstream(),
                )

    # Ten opens that fail after their slot is taken.
    for _ in range(10):
        with pytest.raises(RuntimeError, match="setup failed"):
            evaluation_handle.open(episode(), runtime=FailingSetup(), upstream=scripted_upstream())

    # Exactly eight slots are left, not one fewer for each failure.
    held = hold(DEFAULT_MAX_LIVE_EVALUATIONS)
    try:
        with pytest.raises(LiveEvaluationLimitError) as caught:
            evaluation_handle.open(episode(), upstream=scripted_upstream())
        assert (caught.value.limit, caught.value.live) == (DEFAULT_MAX_LIVE_EVALUATIONS, DEFAULT_MAX_LIVE_EVALUATIONS)
        assert len({port_of(handle) for handle in held}) == DEFAULT_MAX_LIVE_EVALUATIONS
    finally:
        close_all(held)
