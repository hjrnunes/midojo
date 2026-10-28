from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from starlette.datastructures import State

from midojo.types import SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from .config import AppConfig
from .dependencies import rejection_response, require_control_token
from .lifecycle import RejectedError
from .routers import agent, runs, suite, tasks
from .service import ControlPlane
from .store import InMemoryStore, Store

_SUITE_NAME = TypeAdapter(SuiteName)


class _FixedState(State):
    """An app's state whose entries can't be set, replaced or deleted once it is built."""

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__(dict(state))

    def __setattr__(self, key: Any, value: Any) -> None:
        raise AttributeError(f"The app's {key!r} is fixed when the app is built")

    def __delattr__(self, key: Any) -> None:
        raise AttributeError(f"The app's {key!r} is fixed when the app is built")


def create_app(
    suites: Mapping[SuiteName, YAMLTaskSuite],
    *,
    store: Store | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    """The control plane's HTTP app, serving ``suites`` from ``store`` with ``config``.

    Its wiring and auth settings are fixed once it is built. ``app.state``
    holds the control plane (``control``) and, read from it, its ``suites``
    (a read-only mapping), ``store`` and ``config`` (a frozen ``AppConfig``:
    the control token, session lifetime, closure wait and environment
    history); none of those entries can be set, replaced or deleted, and the
    control plane's own ``suites``, ``store``, ``config`` and ``served`` are
    read-only. Every route reads them through the control plane, so the
    control token a route requires is always the one the control plane opens
    host mode under. Code in the app's process can still replace the
    ``app.state`` object itself or the app's dependency overrides, as it can
    any object it holds: that rewires the app rather than configuring it, and
    a route then finds no control plane and fails rather than run open.
    """
    app = FastAPI()
    named = {_SUITE_NAME.validate_python(name): suite for name, suite in suites.items()}
    # Every route that reads or writes an evaluation goes through this one writer. Served, it opens no host-mode
    # evaluation without a control token, whatever path asks.
    control = ControlPlane(named, store=store if store is not None else InMemoryStore(), config=config, served=True)
    app.state = _FixedState(
        {"control": control, "suites": control.suites, "store": control.store, "config": control.config}
    )
    # /agent routes take an evaluation session token instead.
    control = [Depends(require_control_token)]
    app.include_router(suite.router, dependencies=control)
    app.include_router(tasks.router, dependencies=control)
    app.include_router(runs.router, dependencies=control)
    app.include_router(agent.router)

    @app.exception_handler(RejectedError)
    async def rejected(request: Request, exc: RejectedError) -> JSONResponse:
        return rejection_response(exc)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
