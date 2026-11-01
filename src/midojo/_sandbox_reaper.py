"""Remove a run's OpenShell workspace after its driver is gone.

:func:`midojo.episode_handle.open` starts this as a detached process for a
disposable runtime. It waits for the driver process to exit and then deletes
the sandboxes in the driver's workspace and the workspace itself, so a driver
killed with ``SIGKILL`` leaves no sandbox behind. A normal close stops the
reaper before it acts, because the runtime's own teardown already deleted
them.

Run it directly; it is not an API::

    python -m midojo._sandbox_reaper --workspace NAME --driver PID [--cluster NAME]

The workspace names the run's sandbox home, not an evaluation: a deleted
workspace takes every sandbox this driver made with it, and a workspace the
driver never filled is deleted as cleanly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# How often to ask whether the driver is still there.
_POLL_SECONDS = 0.5

# How long to wait for the driver before giving up: a driver that outlives this is treated as gone.
_MAX_WAIT_SECONDS = 24 * 60 * 60

# How long the gateway gets to remove the sandboxes and the workspace once the driver is gone.
_REAP_SECONDS = 60.0

# How long a single sandbox deletion waits to be visible as gone.
_DELETE_SECONDS = 30.0


def _alive(pid: int) -> bool:
    """Whether ``pid`` is still a process (a zombie counts as alive until its parent reaps it)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_driver(pid: int, *, budget: float = _MAX_WAIT_SECONDS) -> bool:
    """Wait for ``pid`` to exit; return whether it did, within ``budget``."""
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(_POLL_SECONDS)
    return False


def reap(workspace: str, *, cluster: str = "", budget: float = _REAP_SECONDS) -> list[str]:
    """Delete every sandbox in ``workspace`` and then the workspace; return what was deleted.

    Deleting a workspace that still holds a sandbox is refused (there is no
    cascade), so the sandboxes go first. A workspace the gateway no longer
    knows is not an error: nothing of the driver's is left.
    """
    from openshell import SandboxClient, WorkspaceClient

    client = SandboxClient.from_active_cluster(cluster=cluster or None, timeout=60.0)
    workspaces = WorkspaceClient.from_sandbox_client(client)
    deleted: list[str] = []
    deadline = time.monotonic() + budget
    try:
        while time.monotonic() < deadline:
            try:
                names = [ref.name for ref in client.list_all(workspace=workspace)]
            except Exception:
                # A workspace the gateway has already removed leaves nothing to reap.
                return deleted
            if not names:
                break
            for name in names:
                try:
                    client.delete(name, workspace=workspace, allow_missing=True)
                except Exception:
                    continue
                try:
                    client.wait_deleted(name, workspace=workspace, timeout_seconds=_DELETE_SECONDS)
                except Exception:
                    pass
                deleted.append(name)
            time.sleep(_POLL_SECONDS)
        try:
            workspaces.delete(workspace, allow_missing=True)
        except Exception:
            pass
    finally:
        try:
            client.close()
        except Exception:
            pass
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="midojo._sandbox_reaper", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--workspace", required=True, help="the run's OpenShell workspace to remove")
    parser.add_argument("--driver", required=True, type=int, help="the process whose exit frees the workspace")
    parser.add_argument("--cluster", default="", help="the OpenShell gateway name; empty for the active one")
    args = parser.parse_args(argv)
    if not _wait_for_driver(args.driver):
        return 0
    try:
        reap(args.workspace, cluster=args.cluster)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
