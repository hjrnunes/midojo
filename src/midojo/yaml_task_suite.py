from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from midojo.attacks import resolve_source, wrap_payload
from midojo.env_inference import infer_environment_type
from midojo.observations import evidence_status
from midojo.probes import substitute_probes, substitute_probes_deep
from midojo.runtimes import AgentRuntime, build_runtime
from midojo.runtimes.openshell import OBSERVATIONS_SOURCE, OpenShellObservations
from midojo.suite_definition import OpenShellRuntimeDefinition, ProbeDefinition, SuiteDefinition
from midojo.types import CheckValidity, Environment, FunctionCallRecord, GradeValidity, SuiteName, ValidityReason
from midojo.verifier import Check, VerificationContext, VerificationResult, parse_check
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


def _validity(result: VerificationResult) -> CheckValidity:
    """A grade is invalid exactly when its top-level check resolved to unknown."""
    if result.passed is not None:
        return CheckValidity(status="valid")
    reasons = list(result.unknowns) or [ValidityReason(check=result.reason, reason="the check returned no verdict")]
    return CheckValidity(status="invalid", reasons=reasons)


@dataclass
class UserTask:
    id: str
    prompt: str
    check: Check


@dataclass
class InjectionTask:
    id: str
    description: str
    probes: dict[str, str] = field(default_factory=dict)
    check: Check | None = None


class YAMLTaskSuite:
    """Reads a suite definition from a single suite.yaml file."""

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
        self.runtime: AgentRuntime = build_runtime(self.name, self.definition.agent_runtime)
        self.environment_type: type[Environment] = infer_environment_type(self.name, self.definition.environment)
        self.user_tasks: dict[str, UserTask] = {}
        self.injection_tasks: dict[str, InjectionTask] = {}
        self._register_tasks()

    def provision_environment(self, injections: dict[str, str]) -> Environment:
        """A fresh environment from the declared state, with the active probe payloads substituted."""
        return self.environment_type.model_validate(substitute_probes_deep(self.definition.environment, injections))

    def inject_user_task_prompt(self, user_task_id: str, injections: dict[str, str]) -> str:
        return substitute_probes(self.user_tasks[user_task_id].prompt, injections)

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
    ) -> dict[str, Any]:
        """Grade one evaluation.

        ``utility`` and ``security`` are True, False, or None when the check
        couldn't be decided from the evidence collected; ``grade_validity`` marks
        such a grade ``invalid`` and lists the unknown checks. Each grade has a
        reason, whatever its outcome. Without an injection task, ``security`` is
        False, its reason None, and its validity ``not_applicable``.
        ``evidence_status`` is how completely each evidence source was collected.
        """
        observations = dict(observations or {})
        if self.runtime.name == "openshell" and OBSERVATIONS_SOURCE not in observations:
            observations[OBSERVATIONS_SOURCE] = OpenShellObservations.not_collected()
        ctx = VerificationContext(
            agent_output=agent_output,
            pre_environment=pre_environment,
            post_environment=post_environment,
            function_calls=function_calls,
            observations=observations,
        )

        # One traversal per check grades it and names the criterion that decided it.
        utility = self.user_tasks[user_task_id].check.assess(ctx)

        injection_check = self.injection_tasks[injection_task_id].check if injection_task_id is not None else None
        security: bool | None = False
        security_reason: str | None = None
        security_validity = CheckValidity(status="not_applicable")
        if injection_check is not None:
            result = injection_check.assess(ctx)
            security, security_reason, security_validity = result.passed, result.reason, _validity(result)

        return {
            "utility": utility.passed,
            "utility_reason": utility.reason,
            "security": security,
            "security_reason": security_reason,
            "grade_validity": GradeValidity(utility=_validity(utility), security=security_validity).model_dump(),
            "evidence_status": evidence_status(observations),
        }

    def _register_tasks(self) -> None:
        for task in self.definition.user_tasks:
            check = self._parse_check(task.id, task.utility)
            self.user_tasks[task.id] = UserTask(id=task.id, prompt=task.prompt, check=check)

        for task in self.definition.injection_tasks:
            check = self._parse_check(task.id, task.security)
            probes = self._parse_probes(task.id, task.probes)
            self.injection_tasks[task.id] = InjectionTask(
                id=task.id,
                description=task.description,
                check=check,
                probes=probes,
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
