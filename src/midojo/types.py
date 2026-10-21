import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, StringConstraints

# A suite name is also a URL path segment. Dots support external module paths;
# a leading alphanumeric character rules out the special '.' and '..' segments.
SuiteName = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class Environment(BaseModel):
    """Base class for suite environments."""

    ...


class FunctionCallRecord(BaseModel):
    """A recorded function call execution (function + args + result + env snapshots).

    An unfinished call (``complete`` False) was started and hasn't finished: it
    has no ``result`` and no ``post_environment`` yet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identifies the record within its evaluation: the client's call_id, or a server-assigned UUID.",
    )
    function: str
    args: dict
    result: str | None = None
    error: str | None = None
    blocked: bool = Field(
        default=False,
        description="The tool never ran: the agent's call was blocked, and `result` is what the agent was told instead.",
    )
    complete: bool = Field(
        default=True,
        description="False while the call is started but not finished: it has no result or post_environment yet.",
    )
    timestamp: str
    turn: int | None = Field(
        default=None, description="Index of the turn most recently started when the call was recorded, if any."
    )
    pre_environment: SerializeAsAny[Environment]
    post_environment: SerializeAsAny[Environment] | None = None
