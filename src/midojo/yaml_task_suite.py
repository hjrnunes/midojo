from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from midojo_sdk.injection import (
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionPlan,
    InjectionTarget,
    SelectorSurface,
)
from pydantic import BaseModel, ValidationError

from midojo.attacks import resolve_source, wrap_payload
from midojo.control_plane.config import CONTROL_TOKEN_ENV
from midojo.env_inference import infer_environment_type
from midojo.episode import Episode, UnknownTaskError, freeze
from midojo.probes import PROBE_PLACEHOLDER_RE, substitute_probes, substitute_probes_deep
from midojo.runtimes import AgentRuntime, RuntimeRegistry, build_runtime
from midojo.suite_definition import OpenShellRuntimeDefinition, ProbeDefinition, SuiteDefinition
from midojo.types import Environment, FunctionCallRecord, SuiteName
from midojo.verifier import Check, parse_check
from midojo.verifiers import ConversationTurn
from midojo.verifiers.openshell import reads_openshell_observations

# ``${env.VAR}`` references (OGX-style), expanded from the process environment in
# the runtime settings only (see ``_expand_env_vars``). The ``env.`` namespace
# keeps these from colliding with the bare ``${...}`` shell tokens that appear in
# attack payloads, and mirrors the syntax OGX uses in its distribution configs.
_ENV_VAR_RE = re.compile(r"\$\{env\.([A-Za-z_][A-Za-z0-9_]*)(?:(:=|:\+)([^}]*))?\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${env.VAR}`` references in string leaves.

    Deployment-specific settings (notably the sandbox network policy's control
    plane host, which differs between a local gateway and a cluster) can be
    parameterised from the environment instead of hard-coded. The syntax mirrors
    OGX's distribution configs, with an added bare (required) form:

    - ``${env.VAR:=default}`` — ``default`` when VAR is unset *or empty*
    - ``${env.VAR:+value}``   — ``value`` only when VAR is set and non-empty, else ""
    - ``${env.VAR}``          — required: raises when VAR is unset or empty

    This is applied to the ``agent_runtime`` settings only — never to its seeded
    ``files`` or to attack payloads, which legitimately contain bare ``${...}``
    (e.g. ``${IFS}`` shell obfuscation). The ``env.`` namespace means even a
    stray ``${env.X}`` in a payload would be out of scope here; nothing else
    looks like these tokens.
    """
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(_replace_env_match, value)
    return value


def _replace_env_match(match: re.Match[str]) -> str:
    name, op, arg = match.group(1), match.group(2), match.group(3)
    if name == CONTROL_TOKEN_ENV:
        raise ValueError(
            f"environment variable '{name}' referenced in suite agent_runtime: "
            "the control token must not be passed to the agent runtime"
        )
    value = os.environ.get(name)
    if op == ":=":  # ${env.VAR:=default}
        return value or arg
    if op == ":+":  # ${env.VAR:+value}
        return arg if value else ""
    if value:  # bare ${env.VAR} — required
        return value
    raise ValueError(
        f"environment variable '{name}' referenced in suite agent_runtime is unset "
        f"or empty (use '${{env.{name}:=default}}' to provide a default)"
    )


@dataclass
class UserTask:
    id: str
    # The user's messages, one per turn; a prompt task has one.
    turns: list[str]
    check: Check
    # The utility check as authored, frozen: what an episode of the task is graded with.
    spec: Mapping[str, Any] | None = None

    @property
    def prompt(self) -> str:
        return self.turns[0]


@dataclass
class InjectionTask:
    id: str
    description: str
    probes: dict[str, str] = field(default_factory=dict)
    check: Check | None = None
    # The security check as authored, frozen: what an episode of the task is graded with.
    spec: Mapping[str, Any] | None = None


def _template_instructions(
    value: Any, channel: Channel, active: Mapping[str, str], path: tuple[str | int, ...] = ()
) -> list[InjectionInstruction]:
    """Describe every active placeholder occurrence without using provenance to select it."""
    instructions: list[InjectionInstruction] = []
    if isinstance(value, str):
        keys = (f"{match.group(1)}:{match.group(2)}" for match in PROBE_PLACEHOLDER_RE.finditer(value))
        for key in keys:
            if key in active:
                target = InjectionTarget(surface=SelectorSurface.TEMPLATE, template_path=path, template_key=key)
                instructions.append(
                    InjectionInstruction(
                        channel=channel, payload=active[key], target=target, mode=InjectionMode.REPLACE, probe_key=key
                    )
                )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            instructions.extend(_template_instructions(item, channel, active, (*path, key)))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            instructions.extend(_template_instructions(item, channel, active, (*path, index)))
    return instructions


class YAMLTaskSuite:
    """Reads a suite definition from a single suite.yaml file.

    A suite is a catalogue: :meth:`episode` compiles one of its user task x
    injection task pairs into an :class:`~midojo.episode.Episode`. Parsing it
    builds no runtime and imports no runtime provider: the suite records the
    runtime definition its ``agent_runtime`` asks for, and :meth:`new_runtime`
    builds a fresh runtime from it each time.
    """

    def __init__(self, name: SuiteName, suite_yaml_path: Path) -> None:
        self._suite_yaml_path = suite_yaml_path
        raw = yaml.safe_load(suite_yaml_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid suite definition in {suite_yaml_path}: expected a YAML mapping")
        if "name" in raw and raw["name"] != name:
            raise ValueError(f"Suite name in {suite_yaml_path} does not match the supplied name {name!r}")
        runtime = raw.get("agent_runtime")
        if isinstance(runtime, dict):
            raw["agent_runtime"] = {k: v if k == "files" else _expand_env_vars(v) for k, v in runtime.items()}
        try:
            self.definition = SuiteDefinition.model_validate({**raw, "name": name})
        except ValidationError as exc:
            raise ValueError(f"Invalid suite definition in {suite_yaml_path}:\n{exc}") from exc
        self.name = self.definition.name
        self.environment_type: type[Environment] = infer_environment_type(self.name, self.definition.environment)
        self.user_tasks: dict[str, UserTask] = {}
        self.injection_tasks: dict[str, InjectionTask] = {}
        self._register_tasks()

    @property
    def suite_yaml_path(self) -> Path:
        """The suite.yaml this suite was loaded from."""
        return self._suite_yaml_path

    def new_runtime(self, *, registry: RuntimeRegistry | None = None) -> AgentRuntime:
        """A fresh runtime for the suite's ``agent_runtime``: it shares no state with any other.

        It is built by ``registry`` (the default runtime registry when None);
        see :func:`midojo.runtimes.build_runtime`.
        """
        return build_runtime(self.name, self.definition.agent_runtime, registry=registry)

    def episode(
        self,
        user_task_id: str,
        injection_task_id: str | None = None,
        injections: Mapping[str, str] | None = None,
    ) -> Episode:
        """Compile a user task x injection task pair into an episode, with ``injections`` substituted.

        Raises :class:`~midojo.episode.UnknownTaskError` for a task ID the suite
        doesn't have. ``injections`` maps ``"<task>:<probe>"`` to a payload (see
        :meth:`get_probes_for_task`); without it, no probe is active.
        """
        if user_task_id not in self.user_tasks:
            raise UnknownTaskError(f"Unknown user task: {user_task_id}")
        if injection_task_id is not None and injection_task_id not in self.injection_tasks:
            raise UnknownTaskError(f"Unknown injection task: {injection_task_id}")
        active = dict(injections or {})
        injection_task = self.injection_tasks[injection_task_id] if injection_task_id is not None else None
        instructions = _template_instructions(self.definition.environment, Channel.ENVIRONMENT_STATE, active)
        instructions += _template_instructions(self.user_tasks[user_task_id].turns, Channel.USER_PROMPT, active)
        files = getattr(self.definition.agent_runtime, "files", {})
        instructions += _template_instructions(files, Channel.RUNTIME_FILE, active)
        plan = InjectionPlan(tuple(instructions), probe_payloads=tuple(active.items()))
        substitutions = dict(plan.substitution_payloads)
        return Episode(
            runtime=self.definition.agent_runtime,
            turns=self.inject_user_task_turns(user_task_id, substitutions),
            plan=plan,
            security=injection_task.spec if injection_task is not None else None,
            utility=self.user_tasks[user_task_id].spec,
            environment=cast(dict[str, Any], substitute_probes_deep(self.definition.environment, substitutions)),
            environment_type=self.environment_type,
            suite_name=self.name,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
        )

    def provision_environment(self, injections: dict[str, str]) -> Environment:
        """A fresh environment from the declared state, with the active probe payloads substituted."""
        return self.environment_type.model_validate(substitute_probes_deep(self.definition.environment, injections))

    def inject_user_task_prompt(self, user_task_id: str, injections: dict[str, str]) -> str:
        return self.inject_user_task_turns(user_task_id, injections)[0]

    def inject_user_task_turns(self, user_task_id: str, injections: dict[str, str]) -> list[str]:
        return [substitute_probes(message, injections) for message in self.user_tasks[user_task_id].turns]

    def get_probes_for_task(self, task_id: str) -> dict[str, str]:
        probes = self.injection_tasks[task_id].probes
        return {f"{task_id}:{probe_id}": payload for probe_id, payload in probes.items()}

    def grade(
        self,
        user_task_id: str,
        injection_task_id: str | None,
        agent_output: str,
        pre_environment: Environment,
        post_environment: Environment,
        function_calls: list[FunctionCallRecord],
        observations: dict[str, BaseModel] | None = None,
        turns: list[ConversationTurn] | None = None,
    ) -> dict[str, Any]:
        """Grade one evaluation with the pair's checks (see :meth:`Episode.grade <midojo.episode.Episode.grade>`).

        ``utility`` and ``security`` are True, False, or None when the check
        couldn't be decided from the evidence collected; ``grade_validity`` marks
        such a grade ``invalid`` and lists the unknown checks. Each grade has a
        reason, whatever its outcome. Without an injection task, ``security`` is
        False, its reason None, and its validity ``not_applicable``.
        ``evidence_status`` is how completely each evidence source was collected.
        """
        episode = self.episode(user_task_id, injection_task_id)
        return episode.grade(agent_output, pre_environment, post_environment, function_calls, observations, turns)

    def _register_tasks(self) -> None:
        for task in self.definition.user_tasks:
            check = self._parse_check(task.id, task.utility)
            self.user_tasks[task.id] = UserTask(id=task.id, turns=task.messages, check=check, spec=freeze(task.utility))

        for task in self.definition.injection_tasks:
            check = self._parse_check(task.id, task.security)
            probes = self._parse_probes(task.id, task.probes)
            self.injection_tasks[task.id] = InjectionTask(
                id=task.id,
                description=task.description,
                check=check,
                probes=probes,
                spec=freeze(task.security),
            )

    def _parse_check(self, task_id: str, raw: dict[str, Any]) -> Check:
        check = parse_check(raw)
        if not isinstance(self.definition.agent_runtime, OpenShellRuntimeDefinition) and (
            reads_openshell_observations(check.parsed)
        ):
            raise ValueError(f"Task {task_id!r} uses OpenShell predicates, which require the openshell agent_runtime")
        return check

    def _parse_probes(self, task_id: str, definitions: dict[str, ProbeDefinition]) -> dict[str, str]:
        probes: dict[str, str] = {}
        for probe_id, probe in definitions.items():
            try:
                payload = self._resolve_probe_payload(probe)
                probes[probe_id] = wrap_payload(payload, probe.attack_type)
            except ValueError as e:
                raise ValueError(f"Probe '{task_id}:{probe_id}': {e}") from None
        return probes

    def _resolve_probe_payload(self, probe: ProbeDefinition) -> str:
        """A probe's cargo is either an inline ``payload`` or a library ``source``."""
        if probe.payload is not None:
            return probe.payload
        assert probe.source is not None  # SuiteDefinition validated the exclusive choice.
        payload_set = resolve_source(probe.source, base_dir=self._suite_yaml_path.parent)
        index = probe.index
        if not 0 <= index < len(payload_set.payloads):
            raise ValueError(
                f"index {index} out of range for payload set '{payload_set.id}' ({len(payload_set.payloads)} payloads)"
            )
        return payload_set.payloads[index]
