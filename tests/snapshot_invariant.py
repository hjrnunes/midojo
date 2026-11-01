"""The suite-wide check architecture 2.1.1 asks for: one implementation names a snapshot.

A sealed snapshot holds only values JSON carries back as the types it keeps, so its
ID is :func:`midojo_sdk.snapshot.snapshot_digest` of the JSON form ``GET .../snapshot``
serves, and a client reading that form back gets the same ID. This module installs
that check around every ``Snapshot`` the suite builds.

A snapshot holding a value JSON does not carry back (``bytes``, a date, a tuple or
set, a non-finite float) keeps MiDojo's Python-value encoder, which the fallback in
``control_plane.state.snapshot_digest`` documents; ``tests/test_snapshot_digest.py``
covers that path.
"""

from __future__ import annotations

from collections.abc import Callable

from midojo_sdk import snapshot as sdk_snapshot

from midojo.control_plane.routers.runs import snapshot_response
from midojo.control_plane.state import _NOT_JSON, Snapshot, _json_data, _snapshot_content

_deepest = 0


def assert_named_by_the_sdk_digest(snapshot: Snapshot) -> None:
    """Fail unless the SDK names ``snapshot``'s JSON form by the ID the snapshot carries."""
    if _json_data(_snapshot_content(snapshot)) is _NOT_JSON:
        return

    served = snapshot_response(snapshot).model_dump(mode="json")
    expected = sdk_snapshot.snapshot_digest(served)

    assert snapshot.snapshot_id == expected, (
        f"snapshot {snapshot.snapshot_id} is not the SDK digest of the JSON served for it ({expected})"
    )
    assert sdk_snapshot.verified(served) is served


def install() -> Callable[[], None]:
    """Check every snapshot the suite builds. Call what this returns to undo it."""
    global _deepest

    original = Snapshot.__init__

    def checked(self: Snapshot, *args: object, **kwargs: object) -> None:
        global _deepest

        original(self, *args, **kwargs)
        if _deepest:
            return
        _deepest += 1
        try:
            assert_named_by_the_sdk_digest(self)
        finally:
            _deepest -= 1

    Snapshot.__init__ = checked

    def undo() -> None:
        Snapshot.__init__ = original

    return undo
