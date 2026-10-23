from __future__ import annotations

from collections.abc import Mapping

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter

from midojo.types import SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from .config import AppConfig
from .dependencies import rejection_response, require_control_token
from .lifecycle import RejectedError
from .routers import agent, runs, suite, tasks
from .service import ControlPlane
from .store import InMemoryStore, Store

_SUITE_NAME = TypeAdapter(SuiteName)


def create_app(
    suites: Mapping[SuiteName, YAMLTaskSuite],
    *,
    store: Store | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.suites = {_SUITE_NAME.validate_python(name): suite for name, suite in suites.items()}
    app.state.store = store if store is not None else InMemoryStore()
    app.state.config = config if config is not None else AppConfig()
    # Every route that reads or writes an evaluation goes through this one writer.
    app.state.control = ControlPlane(app.state.suites, store=app.state.store, config=app.state.config)
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
