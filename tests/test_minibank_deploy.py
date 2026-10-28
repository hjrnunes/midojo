"""The minibank suite's Kubernetes manifests run it in host mode, and agree with each other and with midojo-run.

Checked offline, on the manifests as written (``suites/minibank/deploy``):
no cluster is reached.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import click
import pytest
import yaml

from midojo import orchestrator

DEPLOY = Path(__file__).parent.parent / "suites" / "minibank" / "deploy"


def _resources() -> list[dict[str, Any]]:
    kustomization = yaml.safe_load((DEPLOY / "kustomization.yaml").read_text())
    return [doc for name in kustomization["resources"] for doc in yaml.safe_load_all((DEPLOY / name).read_text())]


@pytest.fixture(scope="module")
def resources() -> dict[tuple[str, str], dict[str, Any]]:
    return {(doc["kind"], doc["metadata"]["name"]): doc for doc in _resources()}


def _container(workload: dict[str, Any]) -> dict[str, Any]:
    [container] = workload["spec"]["template"]["spec"]["containers"]
    return container


def _env(container: dict[str, Any]) -> dict[str, Any]:
    return {var["name"]: var.get("value", var.get("valueFrom")) for var in container.get("env", [])}


def _midojo_run(job: dict[str, Any]) -> list[str]:
    """The Job's midojo-run command line, as its shell runs it."""
    script = _container(job)["args"][0]
    command = script[script.index("midojo-run") :].replace("\\\n", " ")
    return shlex.split(command)[1:]


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _service_url(resources, url: str) -> tuple[dict[str, Any], int]:
    """The Service a cluster-internal ``http://name:port/...`` URL names, and its port."""
    host, port = url.removeprefix("http://").split("/")[0].split(":")
    service = resources[("Service", host)]
    assert int(port) in [p["port"] for p in service["spec"]["ports"]], url
    return service, int(port)


def _selects(service: dict[str, Any], workload: dict[str, Any]) -> bool:
    labels = workload["spec"]["template"]["metadata"]["labels"]
    return all(labels.get(key) == value for key, value in service["spec"]["selector"].items())


def test_no_fake_mcp_server_is_deployed(resources):
    assert not [key for key in resources if "fake-mcp" in key[1]]
    assert not (DEPLOY / "fake-mcp.yaml").exists()


def test_the_job_runs_midojo_run_in_host_mode_with_options_it_has(resources):
    args = _midojo_run(resources[("Job", "minibank-run-ladder")])
    # midojo-run parses them (an unknown option or value would stop the Job before the run).
    context = orchestrator.main.make_context("midojo-run", list(args))
    assert isinstance(context, click.Context)
    assert context.params["interceptor_port"] == int(_option(args, "--interceptor-port"))
    assert (_option(args, "--interception"), _option(args, "--protocol"), _option(args, "--suite")) == (
        "host",
        "a2a",
        "minibank",
    )
    assert _option(args, "--interceptor-bind") == "0.0.0.0"


def test_the_agent_reaches_the_interceptor_in_the_jobs_pod(resources):
    job = resources[("Job", "minibank-run-ladder")]
    args = _midojo_run(job)
    port = int(_option(args, "--interceptor-port"))
    service = resources[("Service", _option(args, "--interceptor-advertise-host"))]
    assert _selects(service, job)
    assert [(p["port"], p["targetPort"]) for p in service["spec"]["ports"]] == [(port, port)]
    assert {"containerPort": port} in _container(job)["ports"]


def test_the_job_forwards_to_the_real_server_and_drives_the_agent_and_the_control_plane(resources):
    args = _midojo_run(resources[("Job", "minibank-run-ladder")])
    for option, deployment in [
        ("--upstream-mcp-url", "minibank-real-mcp"),
        ("--agent-uri", "minibank-agent"),
        ("--control-url", "minibank-control-plane"),
    ]:
        service, port = _service_url(resources, _option(args, option))
        workload = resources[("Deployment", deployment)]
        assert _selects(service, workload), option
        assert {"containerPort": port} in _container(workload)["ports"], option


def test_the_control_token_is_shared_by_the_control_plane_and_the_job_only(resources):
    token = {"secretKeyRef": {"name": "minibank-control-token", "key": "MIDOJO_CONTROL_TOKEN"}}
    assert _env(_container(resources[("Deployment", "minibank-control-plane")]))["MIDOJO_CONTROL_TOKEN"] == token
    assert _env(_container(resources[("Job", "minibank-run-ladder")]))["MIDOJO_CONTROL_TOKEN"] == token
    example = yaml.safe_load((DEPLOY / "control-token-secret.example.yaml").read_text())
    assert (example["metadata"]["name"], list(example["stringData"])) == (
        "minibank-control-token",
        ["MIDOJO_CONTROL_TOKEN"],
    )


def test_the_agent_gets_nothing_of_the_evaluation_or_the_tools_behind_the_interceptor(resources):
    agent = _container(resources[("Deployment", "minibank-agent")])
    assert set(_env(agent)) == {"AGENT_PUBLIC_URL"}
    assert [source["secretRef"]["name"] for source in agent["envFrom"]] == ["minibank-llm-creds"]
    assert agent["command"] == ["minibank-a2a-agent", "--host", "0.0.0.0"]


def test_kubernetes_gives_the_agent_no_service_addresses(resources):
    # With service links (the default), Kubernetes would set MINIBANK_CONTROL_PLANE_SERVICE_HOST,
    # MINIBANK_REAL_MCP_PORT and the like in the agent's environment: the addresses it must not be given.
    pod = resources[("Deployment", "minibank-agent")]["spec"]["template"]["spec"]
    assert pod["enableServiceLinks"] is False


@pytest.mark.parametrize("deployment,port", [("minibank-control-plane", 8080), ("minibank-real-mcp", 8083)])
def test_only_the_job_reaches_the_control_plane_and_the_real_server(resources, deployment, port):
    workload = resources[("Deployment", deployment)]
    [policy] = [
        doc
        for (kind, _), doc in resources.items()
        if kind == "NetworkPolicy"
        and all(
            workload["spec"]["template"]["metadata"]["labels"].get(k) == v
            for k, v in doc["spec"]["podSelector"]["matchLabels"].items()
        )
    ]
    [rule] = policy["spec"]["ingress"]
    job_labels = resources[("Job", "minibank-run-ladder")]["spec"]["template"]["metadata"]["labels"]
    assert [peer["podSelector"]["matchLabels"] for peer in rule["from"]] == [
        {"app.kubernetes.io/component": job_labels["app.kubernetes.io/component"]}
    ]
    assert rule["ports"] == [{"port": port}]
    agent_labels = resources[("Deployment", "minibank-agent")]["spec"]["template"]["metadata"]["labels"]
    assert agent_labels["app.kubernetes.io/component"] != job_labels["app.kubernetes.io/component"]
