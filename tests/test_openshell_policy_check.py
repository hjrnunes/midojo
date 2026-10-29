"""The OpenShell runtime's check that the suite's network policy allows the endpoint the sandbox must reach.

It reads each ``networkPolicies`` rule's endpoints as OpenShell parses them (host
in any case, with or without one trailing dot; a port as an int or a decimal
string, or in ``ports``), reports the binaries a matching rule is restricted to,
and only warns on a miss unless the suite sets ``require_policy_allows``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import midojo.runtimes.openshell as openshell_runtime
from midojo.runtimes import build_runtime
from midojo.runtimes.openshell import OpenShellRuntime, PolicyCheckError, _policy_rules_allowing
from midojo.suite_definition import OpenShellRuntimeDefinition

HOST = "host.openshell.internal"
INTERCEPTOR_URL = f"http://{HOST}:8766/mcp"


def _policy(*endpoints, binaries=None) -> dict:
    rule: dict = {"endpoints": list(endpoints)}
    if binaries is not None:
        rule["binaries"] = [{"path": path} for path in binaries]
    return {"networkPolicies": {"tools": rule}}


@pytest.mark.parametrize(
    "endpoint",
    [
        {"host": HOST, "port": 8766},
        {"host": "Host.OpenShell.Internal", "port": 8766},
        {"host": f"{HOST}.", "port": 8766},
        {"host": "HOST.OPENSHELL.INTERNAL.", "port": "8766"},
        {"host": HOST, "port": "8766"},
        {"host": HOST, "ports": [8765, 8766]},
        {"host": HOST, "port": 443, "ports": ["8766"]},
    ],
)
def test_an_endpoint_allows_the_host_in_any_case_and_the_port_as_an_int_or_a_decimal_string(endpoint):
    assert _policy_rules_allowing(_policy(endpoint), HOST, 8766) == [("tools", [])]


def test_the_wanted_host_is_normalized_too():
    assert _policy_rules_allowing(_policy({"host": HOST, "port": 8766}), f"{HOST.upper()}.", 8766)


@pytest.mark.parametrize(
    "endpoint",
    [
        {"host": HOST, "port": 8765},
        {"host": HOST, "port": "8765"},
        {"host": HOST, "ports": [8765]},
        {"host": "other.internal", "port": 8766},
        {"host": f"{HOST}..", "port": 8766},
        {"host": f"tools.{HOST}", "port": 8766},
        {"host": HOST},
    ],
)
def test_another_host_or_port_does_not_match(endpoint):
    assert _policy_rules_allowing(_policy(endpoint), HOST, 8766) == []


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"networkPolicies": None},
        {"networkPolicies": ["tools"]},
        {"networkPolicies": {"tools": "endpoints"}},
        {"networkPolicies": {"tools": {"endpoints": {"host": HOST, "port": 8766}}}},
        _policy("host.openshell.internal:8766"),
        _policy(None),
        _policy([HOST, 8766]),
        _policy({"host": 8766, "port": 8766}),
        _policy({"host": None, "port": 8766}),
        _policy({"host": HOST, "port": True}),
        _policy({"host": HOST, "ports": [True]}),
        _policy({"host": HOST, "port": "eighty"}),
        _policy({"host": HOST, "port": " 8766"}),
        _policy({"host": HOST, "port": "+8766"}),
        _policy({"host": HOST, "port": "8766.0"}),
        _policy({"host": HOST, "port": "\uff18\uff17\uff16\uff16"}),  # fullwidth digits
        _policy({"host": HOST, "port": "\u0668766"}),
        _policy({"host": HOST, "port": None}),
        _policy({"host": HOST, "port": [8766]}),
        _policy({"host": HOST, "ports": "8766"}),
    ],
)
def test_a_malformed_entry_never_raises_and_never_matches(policy):
    assert _policy_rules_allowing(policy, HOST, 8766) == []


def test_a_malformed_entry_does_not_hide_a_well_formed_one_beside_it():
    policy = _policy("garbage", {"host": HOST, "port": "x"}, {"host": HOST, "port": 8766})
    assert _policy_rules_allowing(policy, HOST, 8766) == [("tools", [])]


def test_the_proto_field_name_network_policies_is_read_too():
    policy = {"network_policies": {"tools": {"endpoints": [{"host": HOST, "port": 8766}]}}}
    assert _policy_rules_allowing(policy, HOST, 8766) == [("tools", [])]


def test_each_matching_rule_comes_with_the_binaries_it_lists():
    policy = {
        "networkPolicies": {
            "node_only": {"endpoints": [{"host": HOST, "port": 8766}], "binaries": [{"path": "/usr/bin/node"}]},
            "malformed_binaries": {
                "endpoints": [{"host": HOST, "port": 8766}],
                "binaries": [{"path": "/usr/bin/pi"}, "/usr/bin/curl", {"path": 3}, {}],
            },
            "elsewhere": {"endpoints": [{"host": "api.example.com", "port": 443}], "binaries": [{"path": "/x"}]},
        }
    }
    assert _policy_rules_allowing(policy, HOST, 8766) == [
        ("node_only", ["/usr/bin/node"]),
        ("malformed_binaries", ["/usr/bin/pi"]),
    ]


def _start(monkeypatch, runtime: OpenShellRuntime) -> MagicMock:
    workspaces = MagicMock()
    workspaces.create.side_effect = lambda name, **kw: SimpleNamespace(name=name)
    connect = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(openshell_runtime.SandboxClient, "from_active_cluster", connect)
    monkeypatch.setattr(openshell_runtime.WorkspaceClient, "from_sandbox_client", lambda client: workspaces)
    runtime.start_run("a" * 32, suite_name="shell_suite")
    return connect


def _messages(caplog, level: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == level and r.name == openshell_runtime.__name__]


@pytest.mark.parametrize("port", [8766, "8766"])
def test_start_run_does_not_warn_when_a_rule_allows_the_interceptor_however_its_port_is_written(
    monkeypatch, caplog, port
):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=_policy({"host": f"{HOST.upper()}.", "port": port}))
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.INFO, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    assert _messages(caplog, "WARNING") == []


def test_by_default_a_miss_only_warns(monkeypatch, caplog):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=_policy({"host": HOST, "port": 8765}))
    assert runtime.require_policy_allows is False
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.WARNING, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    [warning] = _messages(caplog, "WARNING")
    assert f"does not allow the host-side interceptor at {HOST}:8766" in warning
    assert runtime.workspace_name


def test_the_binaries_a_matching_rule_is_restricted_to_are_logged(monkeypatch, caplog):
    policy = _policy({"host": HOST, "port": 8766}, binaries=["/usr/bin/pi", "/usr/bin/node"])
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy)
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.INFO, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    assert _messages(caplog, "WARNING") == []
    [info] = [m for m in _messages(caplog, "INFO") if "binaries" in m]
    assert f"host-side interceptor at {HOST}:8766" in info
    assert "tools" in info and "/usr/bin/node, /usr/bin/pi" in info


def test_no_binaries_are_reported_when_a_matching_rule_lists_none(monkeypatch, caplog):
    policy = {
        "networkPolicies": {
            "pi_only": {"endpoints": [{"host": HOST, "port": 8766}], "binaries": [{"path": "/usr/bin/pi"}]},
            "any": {"endpoints": [{"host": HOST, "port": 8766}]},
        }
    }
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy)
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.INFO, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    assert [m for m in _messages(caplog, "INFO") if "binaries" in m] == []


@pytest.mark.parametrize(
    "configuration,policy",
    [
        ({"interceptor_url": INTERCEPTOR_URL}, _policy({"host": HOST, "port": 8765})),
        ({"control_url": "http://localhost:8090"}, _policy({"host": HOST, "port": 8766})),
    ],
)
def test_with_require_policy_allows_a_miss_raises_before_anything_is_created(monkeypatch, configuration, policy):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy, require_policy_allows=True)
    runtime.configure(cluster="test", **configuration)
    connect = MagicMock()
    monkeypatch.setattr(openshell_runtime.SandboxClient, "from_active_cluster", connect)
    with pytest.raises(PolicyCheckError, match="does not allow .* require_policy_allows"):
        runtime.start_run("a" * 32, suite_name="shell_suite")
    connect.assert_not_called()
    assert runtime.workspace_name == ""


@pytest.mark.parametrize(
    "configuration,policy",
    [
        ({"interceptor_url": INTERCEPTOR_URL}, _policy({"host": f"{HOST}.", "port": "8766"})),
        ({"control_url": "http://localhost:8090"}, _policy({"host": HOST, "port": 8090})),
    ],
)
def test_with_require_policy_allows_a_hit_runs(monkeypatch, caplog, configuration, policy):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy, require_policy_allows=True)
    runtime.configure(cluster="test", **configuration)
    with caplog.at_level(logging.WARNING, logger=openshell_runtime.__name__):
        connect = _start(monkeypatch, runtime)
    connect.assert_called_once()
    assert _messages(caplog, "WARNING") == []


@pytest.mark.parametrize("policy", [None, {}])
def test_with_require_policy_allows_no_policy_still_only_warns(monkeypatch, caplog, policy):
    # The image's built-in policy governs then, which can't be inspected: nothing to check.
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy, require_policy_allows=True)
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.WARNING, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    [warning] = _messages(caplog, "WARNING")
    assert "No sandbox network policy defined" in warning


def test_with_require_policy_allows_a_sandbox_without_the_session_needs_no_control_plane_rule(monkeypatch):
    runtime = OpenShellRuntime(
        "shell_suite", image="pi", policy=_policy(), inject_session=False, require_policy_allows=True
    )
    runtime.configure(cluster="test", control_url="http://localhost:8090")
    _start(monkeypatch, runtime)
    assert runtime.workspace_name


@pytest.mark.parametrize("required", [True, False])
def test_the_definition_passes_require_policy_allows_through(required):
    runtime = build_runtime("shell_suite", OpenShellRuntimeDefinition(image="pi", require_policy_allows=required))
    assert isinstance(runtime, OpenShellRuntime)
    assert runtime.require_policy_allows is required


def test_require_policy_allows_defaults_to_false_and_must_be_a_bool():
    assert OpenShellRuntimeDefinition(image="pi").require_policy_allows is False
    with pytest.raises(ValidationError):
        OpenShellRuntimeDefinition.model_validate({"image": "pi", "require_policy_allows": "yes"})
