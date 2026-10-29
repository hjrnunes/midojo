"""Validated suite data, separate from runtime construction and evaluation.

Environment state and verifier arguments remain extensible. Their
implementations validate the parts specific to each plugin.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

from midojo.types import SuiteName


class _DefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RuntimeDefinitionModel(_DefinitionModel):
    # What an episode needs to run, not a live runtime: read-only once validated.
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OpenShellRuntimeDefinition(_RuntimeDefinitionModel):
    """Runs the agent in an OpenShell sandbox created for each evaluation."""

    type: Literal["openshell"] = "openshell"
    image: str = Field(min_length=1)
    agent_command: list[str] | None = None
    policy: dict[str, Any] | None = None
    providers: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)
    # Seeded into the sandbox; relative paths land under /sandbox/workdir, absolute paths as given.
    files: dict[str, str] = Field(default_factory=dict)
    # Seconds the agent command may run per evaluation; `midojo-run --agent-timeout` overrides it.
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # Whether an agent-mode sandbox gets the evaluation's session (MIDOJO_SESSION_TOKEN) and the control
    # plane's URL (MIDOJO_URL), for an interception layer in the sandbox. A host-mode sandbox never does.
    inject_session: bool = True
    # Whether a policy without a rule for the endpoint the sandbox must reach (the host-side interceptor, or the
    # control plane for a sandbox with the session) stops the run before any sandbox is created, rather than warns.
    require_policy_allows: bool = False


class UnmanagedRuntimeDefinition(_RuntimeDefinitionModel):
    """The agent runs somewhere MiDojo doesn't control."""

    type: Literal["unmanaged"]


def _runtime_type(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("type", "openshell")
    return getattr(value, "type", None)


AgentRuntimeDefinition = Annotated[
    Annotated[OpenShellRuntimeDefinition, Tag("openshell")] | Annotated[UnmanagedRuntimeDefinition, Tag("unmanaged")],
    Discriminator(_runtime_type),
]


class ProbeDefinition(_DefinitionModel):
    """Exactly one non-null payload or source; empty inline payloads are valid."""

    payload: str | None = None
    source: str | None = None
    index: int = Field(default=0, ge=0)
    attack_type: str = "verbatim"

    @model_validator(mode="after")
    def exactly_one_payload_or_source(self) -> Self:
        if (self.payload is None) == (self.source is None):
            raise ValueError("exactly one of 'payload' or 'source' is required")
        return self


class UserTaskDefinition(_DefinitionModel):
    """Exactly one of a single ``prompt`` or a conversation of ``turns``, sent one at a time."""

    id: str = Field(min_length=1)
    prompt: str | None = None
    turns: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)] | None = None
    utility: dict[str, Any]

    @model_validator(mode="after")
    def exactly_one_prompt_or_turns(self) -> Self:
        # Check the keys given, not their values, so an explicit null counts as given.
        given = sorted({"prompt", "turns"} & self.model_fields_set)
        if len(given) != 1:
            raise ValueError("exactly one of 'prompt' or 'turns' is required")
        if getattr(self, given[0]) is None:
            raise ValueError(f"'{given[0]}' must not be null")
        return self

    @property
    def messages(self) -> list[str]:
        """The user's messages in order: the prompt alone, or each turn."""
        if self.turns is not None:
            return list(self.turns)
        assert self.prompt is not None  # exactly_one_prompt_or_turns guarantees one of them.
        return [self.prompt]


class InjectionTaskDefinition(_DefinitionModel):
    id: str = Field(min_length=1)
    description: str
    probes: dict[str, ProbeDefinition] = Field(default_factory=dict)
    security: dict[str, Any]


class SuiteDefinition(_DefinitionModel):
    """Suite name plus the contents of suite.yaml, validated before setup."""

    name: SuiteName
    agent_runtime: AgentRuntimeDefinition
    environment: dict[str, Any] = Field(default_factory=dict)
    user_tasks: list[UserTaskDefinition] = Field(default_factory=list)
    injection_tasks: list[InjectionTaskDefinition] = Field(default_factory=list)

    @field_validator("user_tasks", "injection_tasks")
    @classmethod
    def unique_task_ids(
        cls, tasks: list[UserTaskDefinition] | list[InjectionTaskDefinition]
    ) -> list[UserTaskDefinition] | list[InjectionTaskDefinition]:
        seen: set[str] = set()
        for task in tasks:
            if task.id in seen:
                raise ValueError(f"duplicate task ID: {task.id!r}")
            seen.add(task.id)
        return tasks
