from __future__ import annotations

import logging

import click
import uvicorn

from midojo.control_plane.config import CONTROL_TOKEN_ENV, AppConfig
from midojo.control_plane.main import create_app
from midojo.suites import get_suite

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--port", default=8080, type=int, help="Port to bind to.")
@click.option(
    "--load-suite",
    multiple=True,
    required=True,
    metavar="NAME_OR_MODULE",
    help="Load a suite at startup by name or module path. Repeat to expose multiple suites.",
)
@click.option(
    "--session-ttl",
    default=AppConfig().session_ttl_seconds,
    type=click.IntRange(min=1),
    help="Evaluation session lifetime in seconds.",
)
@click.option(
    "--closure-wait",
    default=AppConfig().closure_wait_seconds,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Seconds an evaluation's already admitted tool calls may still finish once it starts closing.",
)
@click.option(
    "--environment-history",
    is_flag=True,
    default=False,
    help="Record a snapshot of the environment when each tool call starts and finishes. Off by default.",
)
@click.option(
    "--control-token",
    envvar=CONTROL_TOKEN_ENV,
    default=None,
    help=f"Bearer token required on every route except /agent and /health; env: {CONTROL_TOKEN_ENV}.",
)
def main(
    host: str,
    port: int,
    load_suite: tuple[str, ...],
    session_ttl: int,
    closure_wait: float,
    environment_history: bool,
    control_token: str | None,
) -> None:
    config = AppConfig(
        session_ttl_seconds=session_ttl,
        closure_wait_seconds=closure_wait,
        environment_history=environment_history,
        control_token=control_token or None,
    )
    if config.control_token is None:
        logger.warning(
            "The suite, task and run routes are unauthenticated; set %s to require a control token.",
            CONTROL_TOKEN_ENV,
        )
    suites = {name: get_suite(name) for name in dict.fromkeys(load_suite)}
    app = create_app(suites, config=config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
