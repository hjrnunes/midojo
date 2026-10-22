from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, StringConstraints

# A suite name is also a URL path segment. Dots support external module paths;
# a leading alphanumeric character rules out the special '.' and '..' segments.
SuiteName = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class Environment(BaseModel):
    """Base class for suite environments."""

    ...


class FunctionCallRecord(BaseModel):
    """A recorded function call execution (function + args + result + env snapshots)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    function: str
    args: dict
    result: str
    error: str | None = None
    blocked: bool = Field(
        default=False,
        description="The tool never ran: the agent's call was blocked, and `result` is what the agent was told instead.",
    )
    timestamp: str
    pre_environment: SerializeAsAny[Environment]
    post_environment: SerializeAsAny[Environment]


# A grade is ``valid`` when its check resolved to True or False, ``invalid`` when
# it resolved to unknown (the evidence it needed wasn't completely collected),
# and ``not_applicable`` when there was nothing to grade (no injection task).
Validity = Literal["valid", "invalid", "not_applicable"]


class ValidityReason(BaseModel):
    """A check that resolved to unknown, and why."""

    model_config = ConfigDict(frozen=True)

    check: str
    reason: str


class CheckValidity(BaseModel):
    """The validity of one grade, with the unknown checks that made it invalid."""

    status: Validity
    reasons: list[ValidityReason] = Field(default_factory=list)


class GradeValidity(BaseModel):
    """The validity of an evaluation's utility and security grades."""

    utility: CheckValidity
    security: CheckValidity
