import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, StringConstraints, model_validator

# A suite name is also a URL path segment. Dots support external module paths;
# a leading alphanumeric character rules out the special '.' and '..' segments.
SuiteName = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class Environment(BaseModel):
    """Base class for suite environments."""

    ...


# What the interception layer did with a call: ``forwarded`` to the real tool,
# ``simulated`` by a tool the layer defines, or refused before the tool ran:
# ``denied`` by a check on the call's arguments, or ``blocked`` because the tool
# is blocked for the evaluation. A refused call is a ``blocked`` record.
CallDisposition = Literal["forwarded", "denied", "simulated", "blocked"]
REFUSED_DISPOSITIONS: frozenset[CallDisposition] = frozenset({"denied", "blocked"})


class FunctionCallRecord(BaseModel):
    """A recorded function call: what the agent asked for and what it saw.

    A record is an intercepted request: it isn't proof of what the call did,
    to the simulated environment or to a real system. An unfinished call
    (``complete`` False) was started and hasn't finished: it has no ``result``
    yet.

    ``pre_environment`` and ``post_environment`` are environment history, kept
    only when the control plane runs with it on: the environment observed when
    the call started and when it finished. They are observations over the
    call's interval, not its effect: another call running at the same time, or
    the host, may have made a change between them, and an unfinished call has
    no ``post_environment``.
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
    disposition: CallDisposition | None = Field(
        default=None,
        description=(
            "What the interception layer did with the call: forwarded, simulated, denied or blocked. A denied or "
            "blocked call is `blocked`. None when the record doesn't say, as in records made before the field."
        ),
    )
    complete: bool = Field(
        default=True,
        description="False while the call is started but not finished: it has no result yet.",
    )
    timestamp: str
    turn: int | None = Field(
        default=None, description="Index of the turn most recently started when the call was recorded, if any."
    )
    pre_environment: SerializeAsAny[Environment] | None = Field(
        default=None,
        description=(
            "Environment history, when kept: the environment observed when the call started (for a call recorded "
            "whole, when the previous call finished). An observation, not a cause."
        ),
    )
    post_environment: SerializeAsAny[Environment] | None = Field(
        default=None,
        description=(
            "Environment history, when kept: the environment observed when the call finished. Changes since "
            "pre_environment happened during the call's interval; the call didn't necessarily make them."
        ),
    )

    @model_validator(mode="after")
    def _disposition_matches_blocked(self) -> "FunctionCallRecord":
        if self.disposition is not None and (self.disposition in REFUSED_DISPOSITIONS) != self.blocked:
            raise ValueError(f"A {self.disposition} call must have blocked={not self.blocked}.")
        return self


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
