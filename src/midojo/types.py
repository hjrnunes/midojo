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

# Where an evaluation's tool calls are intercepted, chosen by the host when it
# opens the evaluation: ``agent``, by an interception layer inside or beside the
# target that holds the evaluation's session (an in-agent SDK: PI, LangChain,
# a fake MCP server the target reaches); or ``host``, by an interceptor the host
# runs outside the target, which alone holds the session.
InterceptionMode = Literal["agent", "host"]

# How a call record reached the control plane, set by the control plane from the
# authority the call arrived with, never by the caller: ``host`` from the host
# itself (the run-scoped routes, or in process by ID); ``interceptor`` with the
# session of an evaluation intercepted on the host; ``agent`` with the session of
# an evaluation intercepted in the agent. ``agent`` evidence is lower trust: the
# target may hold the session, so it could have written, or left out, any record.
CallProvenance = Literal["host", "interceptor", "agent"]


# How one injection instruction fared on one call, as the interceptor that placed it reported it: the values of
# ``midojo_sdk.injection.OutcomeStatus``. ``partial`` is a structured-content placement the result's text doesn't
# carry (its text isn't a JSON mirror of the structured content), so an agent reading only the text never saw it.
InjectionStatus = Literal["applied", "partial", "fallback", "unmatched", "no_result", "too_large", "skipped_occurrence"]

# The transformation an interceptor applied to a payload: the values of ``midojo_sdk.injection.Transformation``.
InjectionTransformation = Literal["identity", "json_string"]

# A part of a tool result an interceptor returned: its text (content) blocks or its structured content.
ResultPart = Literal["content_block", "structured_content"]


class CallInjectionOutcome(BaseModel):
    """One injection instruction's outcome on a recorded call, as the interceptor reported it in the call's finish.

    It names the instruction by its index in the plan the interceptor was
    given, never the payload or the interceptor's counter scope.

    It is also the instruction's delivery fact for the call: what the
    interceptor ``selected``, how it ``transformation``-ed the payload, and
    which parts of the result it returned it ``emitted`` the payload in, at
    its own boundary. The interceptor records a result before it returns it,
    so a fact never says what the agent, or its model, saw. Each is None when
    the layer didn't report it (a record made before the fields).
    """

    instruction_index: int = Field(ge=0, description="The instruction's index in the interceptor's plan.")
    probe_key: str | None = Field(default=None, description="The instruction's probe, when it names one.")
    occurrence: int = Field(
        ge=0,
        description=(
            "Which matching call of the instruction's this was, one-based; 0 for a call whose arguments didn't "
            "match the instruction's `call_arguments` (`unmatched`), which isn't counted."
        ),
    )
    status: InjectionStatus
    detail: str = Field(description="Why: the selector that matched or didn't, or what was left unchanged.")
    selected: bool | None = Field(
        default=None,
        description="Whether the instruction's exact selector matched a place in the call's result. None: not reported.",
    )
    transformation: InjectionTransformation | None = Field(
        default=None,
        description=(
            "The transformation applied to the payload. None when it was never transformed (its occurrence rule "
            "skipped the call, the call's arguments didn't match its `call_arguments`, or there was no result) or "
            "the layer didn't report it."
        ),
    )
    emitted: list[ResultPart] | None = Field(
        default=None,
        description=(
            "The parts of the result the interceptor returned that carry the transformed payload where it placed "
            "it, checked after every instruction for the call. Empty when none does. None: not reported."
        ),
    )


class FunctionCallRecord(BaseModel):
    """A recorded function call: what the agent asked for and what the interception layer returned to it.

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
        description=(
            "The tool never ran: the agent's call was blocked, and `result` is what the layer returned instead."
        ),
    )
    disposition: CallDisposition | None = Field(
        default=None,
        description=(
            "What the interception layer did with the call: forwarded, simulated, denied or blocked. A denied or "
            "blocked call is `blocked`. None when the record doesn't say, as in records made before the field."
        ),
    )
    structured_content: dict | None = Field(
        default=None,
        description=(
            "The structured content the interception layer returned to the agent with `result` (its text), as it "
            "returned it, when the layer reported the call's details (`injection_outcomes` is a list). Recorded "
            "before it was returned: not proof the agent received it. None when the result had none, the call "
            "is unfinished, or the layer reported its text only."
        ),
    )
    injection_outcomes: list[CallInjectionOutcome] | None = Field(
        default=None,
        description=(
            "How each injection instruction that matched the call fared, as the interception layer reported it: "
            "the instructions it admitted for the call, those whose `call_arguments` the call's arguments didn't "
            "match (`unmatched`, occurrence 0), and those whose occurrence rule skipped it "
            "(`skipped_occurrence`); `partial` when only `structured_content` carries the payload. Empty when no "
            "instruction matched. None when the layer recorded the call's text only (a plain finish, a layer or "
            "recorder without details, or a record made before the field): the record then doesn't say whether "
            "the result had structured content or an injection was placed in it."
        ),
    )
    complete: bool = Field(
        default=True,
        description="False while the call is started but not finished: it has no result yet.",
    )
    provenance: CallProvenance | None = Field(
        default=None,
        description=(
            "How the record reached the control plane, set by the control plane: host, interceptor (a host-side "
            "interceptor) or agent (an in-agent SDK holding the session: lower trust). None in older records."
        ),
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
