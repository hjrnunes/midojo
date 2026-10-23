"""Configuration for a control-plane application instance."""

from pydantic import BaseModel, Field

# Environment variable that sets the control token for midojo-serve and midojo-run.
CONTROL_TOKEN_ENV = "MIDOJO_CONTROL_TOKEN"

# How long closure waits for admitted calls to finish, by default.
DEFAULT_CLOSURE_WAIT_SECONDS = 5.0


class AppConfig(BaseModel):
    session_ttl_seconds: int = Field(default=3600, gt=0)
    # Bearer token required on every route except /agent and /health. None leaves them open.
    control_token: str | None = Field(default=None, min_length=1)
    # Once an evaluation starts closing, how long calls already admitted may still finish.
    closure_wait_seconds: float = Field(default=DEFAULT_CLOSURE_WAIT_SECONDS, ge=0)
    # Whether call records carry environment snapshots (FunctionCallRecord.pre_environment and post_environment):
    # what the environment was when each call started and finished. Off by default: they show what changed during
    # a call, not what the call caused, and each one copies the whole environment.
    environment_history: bool = False
