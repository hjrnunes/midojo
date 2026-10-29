"""The OpenShell runtime's observations: the evidence it collects outside the agent, as models.

:class:`~midojo.runtimes.openshell.OpenShellRuntime` produces them, the
orchestrator records them as ``observations["openshell"]``, and the predicates
in :mod:`midojo.verifiers.openshell` grade them. They are kept apart from the
runtime so that reading, validating and grading them needs none of the
runtime's dependencies (gRPC, protobuf and the OpenShell SDK): this module
imports none of them, and the runtime imports the models from here. Importing
it registers the ``openshell`` observation source
(:func:`midojo.observations.register_observation_type`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from midojo.observations import EvidenceStatus, register_observation_type

if TYPE_CHECKING:
    from midojo.runtimes.openshell.logs import HttpEvent

# Recorded as ``policy_digest`` when the sandbox spec carries no policy, so the
# image's built-in policy governs the sandbox and there is nothing to hash.
BUILTIN_POLICY_DIGEST = "builtin"

# Key of the OpenShell entry in an evaluation's runtime observations.
OBSERVATIONS_SOURCE = "openshell"


class ProcessRecord(BaseModel):
    """A process OCSF saw launch in the sandbox."""

    model_config = ConfigDict(extra="forbid")

    binary: str
    pid: int
    command: str | None = None  # the launched command line, when OCSF logged it
    exit_code: int | None = None  # None when no PROC:TERMINATE event was logged


class HttpRequestRecord(BaseModel):
    """An HTTP request OpenShell's policy proxy inspected (L7), from one OCSF ``HTTP:`` line.

    ``raw`` is the OCSF message the record was parsed from, as logged, and
    ``url`` the URL in it; ``host``, ``port``, ``path`` and ``query`` are derived
    from the URL (:func:`midojo.runtimes.openshell.logs.split_url`), each None
    where it has no such part or couldn't be split. ``raw`` has no default: a
    record built by hand (a test, a replay) passes the line it stands for, a
    synthesized one such as ``HTTP:GET [INFO] ALLOWED GET http://example.com/``
    if it has no logged one.
    """

    model_config = ConfigDict(extra="forbid")

    method: str  # upper-case, e.g. "POST"
    url: str  # as logged, query string included
    decision: Literal["allowed", "denied"]  # the proxy's ALLOWED or DENIED
    host: str | None = None  # lower-case
    port: int | None = None  # only when the URL names one
    path: str | None = None  # "/" for a URL with no path
    query: str | None = None  # without the "?"; None when empty
    raw: str  # the OCSF message the record was parsed from

    @classmethod
    def of(cls, event: HttpEvent) -> HttpRequestRecord:
        """The record of a parsed ``HTTP:`` line."""
        return cls(
            method=event.method,
            url=event.url,
            decision="allowed" if event.action == "ALLOWED" else "denied",
            host=event.host,
            port=event.port,
            path=event.path,
            query=event.query,
            raw=event.raw,
        )


class OpenShellEvidence(BaseModel):
    """How completely each part of the OpenShell evidence was collected.

    Defaults to ``complete``, so observations recorded by a producer that doesn't
    report status are graded as before.
    """

    model_config = ConfigDict(extra="forbid")

    ocsf: EvidenceStatus = "complete"  # processes, network calls, HTTP requests, findings
    workdir: EvidenceStatus = "complete"  # files created, modified and deleted, and new files' contents


class OpenShellObservations(BaseModel):
    """What OpenShell saw the agent do during one evaluation.

    Collected by ``OpenShellRuntime.observe()`` outside the agent, and recorded
    by the orchestrator as ``observations["openshell"]``.
    """

    model_config = ConfigDict(extra="forbid")

    # Fields added after snapshots were first digested: the snapshot digest
    # leaves each out while it's empty (observations.digest_content), so a
    # snapshot recorded without one keeps its ID.
    DIGEST_OMITS_WHEN_EMPTY: ClassVar[tuple[str, ...]] = ("http_requests",)

    evidence_status: OpenShellEvidence = Field(default_factory=OpenShellEvidence)

    # File diff against the seeded files (absolute paths). Created files are
    # found in the workdir only; seeded files are checked wherever they were seeded.
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    workdir_new_file_contents: dict[str, str] = Field(default_factory=dict)

    # OCSF events (kernel-verified)
    processes: list[ProcessRecord] = Field(default_factory=list)  # in launch order
    network_calls_allowed: list[str] = Field(default_factory=list)  # "host:port"
    network_calls_blocked: list[str] = Field(default_factory=list)  # "host:port", or "host" for DNS refusals
    # Closed because the sandbox's policy changed mid-connection, not denied by policy
    network_calls_interrupted: list[str] = Field(default_factory=list)
    # Sandbox-verified executables that made allowed or blocked network calls
    network_callers: list[str] = Field(default_factory=list)
    security_findings: list[str] = Field(default_factory=list)  # finding titles
    # HTTP requests the proxy inspected, allowed and denied, in log order. Part of
    # the ``ocsf`` evidence: none when the log read failed, those read so far when
    # it's incomplete. Empty in observations recorded before it existed.
    http_requests: list[HttpRequestRecord] = Field(default_factory=list)

    # What ran. Optional so evaluations recorded before these fields still validate.
    image: str | None = None  # the image reference the suite gave
    # None with OpenShell 0.1.2, which reports no resolved image digest: the SDK's
    # SandboxRef carries no image at all, and the Sandbox proto (spec and
    # SandboxStatus) carries only the requested reference, SandboxTemplate.image.
    image_digest: str | None = None
    policy_digest: str | None = None  # "sha256:<hex>", or BUILTIN_POLICY_DIGEST

    @classmethod
    def not_collected(cls) -> OpenShellObservations:
        """Stands in for observations that were never recorded."""
        return cls(evidence_status=OpenShellEvidence(ocsf="not_collected", workdir="not_collected"))


register_observation_type(OBSERVATIONS_SOURCE, OpenShellObservations)
