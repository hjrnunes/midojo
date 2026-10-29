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
from google.protobuf.json_format import MessageToDict, ParseError
from openshell._proto import openshell_pb2, sandbox_pb2
from pydantic import ValidationError

import midojo.runtimes.openshell as openshell_runtime
from midojo.runtimes import build_runtime
from midojo.runtimes.openshell import OpenShellRuntime, PolicyCheckError, _policy_rules_allowing, _resolve_policy
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


_FULLWIDTH_8766 = "\uff18\uff17\uff16\uff16"
_ARABIC_INDIC_8 = "\u0668"

# Each policy's network rules as written, against the policy the sandbox gets from them (_resolve_policy).
_POLICY_CASES = [
    # Port spellings protobuf reads as 8766.
    _policy({"host": HOST, "port": 8766}),
    _policy({"host": HOST, "port": "8766"}),
    _policy({"host": HOST, "port": "+8766"}),
    _policy({"host": HOST, "port": "8766.0"}),
    _policy({"host": HOST, "port": 8766.0}),
    _policy({"host": HOST, "port": _FULLWIDTH_8766}),
    _policy({"host": HOST, "port": f"{_ARABIC_INDIC_8}766"}),
    _policy({"host": HOST, "port": "0" * 5000 + "8766"}),
    # Other ports, unset ones and ones protobuf refuses.
    _policy({"host": HOST, "port": 8765}),
    _policy({"host": HOST, "port": "1e3"}),
    _policy({"host": HOST, "port": 0}),
    _policy({"host": HOST, "port": "0"}),
    _policy({"host": HOST, "port": "-0"}),
    _policy({"host": HOST, "port": None}),
    _policy({"host": HOST}),
    _policy({"host": HOST, "port": 65536}),
    _policy({"host": HOST, "port": "70000"}),
    _policy({"host": HOST, "port": -8766}),
    _policy({"host": HOST, "port": 8766 + 2**32}),
    _policy({"host": HOST, "port": 8766.5}),
    _policy({"host": HOST, "port": True}),
    _policy({"host": HOST, "port": False}),
    _policy({"host": HOST, "port": "eighty"}),
    _policy({"host": HOST, "port": " 8766"}),
    _policy({"host": HOST, "port": "0x223e"}),
    _policy({"host": HOST, "port": [8766]}),
    _policy({"host": HOST, "port": "9" * 5000}),
    # ports, in both orders, beside an unset, other or refused port.
    _policy({"host": HOST, "ports": [8765, 8766]}),
    _policy({"host": HOST, "ports": [8766, 8765]}),
    _policy({"host": HOST, "ports": ["8766"]}),
    _policy({"host": HOST, "ports": [0]}),
    _policy({"host": HOST, "port": 0, "ports": [8766]}),
    _policy({"host": HOST, "port": "0", "ports": [8766]}),
    _policy({"host": HOST, "port": None, "ports": [8766]}),
    _policy({"host": HOST, "port": 8766, "ports": None}),
    _policy({"host": HOST, "port": 443, "ports": ["+8766"]}),
    _policy({"host": HOST, "port": 70000, "ports": [8766]}),
    _policy({"host": HOST, "port": 8766, "ports": "garbage"}),
    _policy({"host": HOST, "port": 8766, "ports": 8766}),
    _policy({"host": HOST, "port": 8766, "ports": [8765, "x"]}),
    _policy({"host": HOST, "port": 8766, "ports": [True]}),
    _policy({"host": HOST, "port": "x", "ports": [8766]}),
    _policy({"host": HOST, "port": "9" * 5000, "ports": [8766]}),
    _policy({"host": HOST, "ports": ["9" * 5000]}),
    # Hosts.
    _policy({"host": "Host.OpenShell.Internal", "port": 8766}),
    _policy({"host": f"{HOST}.", "port": "8766"}),
    _policy({"host": f"{HOST}..", "port": 8766}),
    _policy({"host": f"tools.{HOST}", "port": 8766}),
    _policy({"host": "other.internal", "port": 8766}),
    _policy({"host": 8766, "port": 8766}),
    _policy({"host": None, "port": 8766}),
    _policy({"host": HOST, "port": 8766, "unknown": 1}),
    # Rules and endpoints of the wrong shape.
    {"networkPolicies": None},
    {"networkPolicies": ["tools"]},
    {"networkPolicies": {"tools": "endpoints"}},
    {"networkPolicies": {"tools": {"endpoints": {"host": HOST, "port": 8766}}}},
    _policy("host.openshell.internal:8766"),
    _policy(None),
    _policy([HOST, 8766]),
]


def _sandbox_policy_allows(policy: dict, host: str, port: int) -> bool:
    """Whether the policy the sandbox gets allows ``host:port``; one protobuf refuses gives no sandbox at all."""
    spec = openshell_pb2.SandboxSpec()
    try:
        _resolve_policy(policy, spec)
    except ParseError:
        return False
    return any(
        endpoint.host.lower().removesuffix(".") == host and port in {endpoint.port, *endpoint.ports} - {0}
        for rule in spec.policy.network_policies.values()
        for endpoint in rule.endpoints
    )


def test_the_endpoint_parsed_alone_is_the_message_the_sandbox_policy_holds():
    endpoint = openshell_pb2.SandboxSpec().policy.network_policies["tools"].endpoints.add()
    assert type(endpoint) is sandbox_pb2.NetworkEndpoint


@pytest.mark.parametrize("policy", _POLICY_CASES)
def test_the_check_allows_exactly_what_the_sandbox_policy_allows(policy):
    expected = [("tools", [])] if _sandbox_policy_allows(policy, HOST, 8766) else []
    assert _policy_rules_allowing(policy, HOST, 8766) == expected


@pytest.mark.parametrize("policy", _POLICY_CASES)
def test_with_require_policy_allows_the_run_stops_exactly_when_the_sandbox_policy_denies(monkeypatch, policy):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy, require_policy_allows=True)
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    if _sandbox_policy_allows(policy, HOST, 8766):
        _start(monkeypatch, runtime)
        assert runtime.workspace_name
    else:
        with pytest.raises(PolicyCheckError):
            _start(monkeypatch, runtime)


def test_the_policy_cases_cover_both_outcomes():
    outcomes = [_sandbox_policy_allows(policy, HOST, 8766) for policy in _POLICY_CASES]
    assert outcomes.count(True) >= 15 and outcomes.count(False) >= 30


@pytest.mark.parametrize(
    "malformed",
    [
        "garbage",
        {"host": HOST, "port": "x"},
        {"host": HOST, "port": "9" * 5000},
        {"host": HOST, "ports": ["9" * 5000]},
        {"host": HOST, "port": 8766, "ports": "garbage"},
    ],
)
def test_a_malformed_endpoint_does_not_hide_a_well_formed_one_beside_it(malformed):
    # Where the check and the sandbox differ: protobuf refuses the whole policy, so setup() would fail on it
    # anyway; the check reads each endpoint on its own and still finds the well-formed one.
    policy = _policy(malformed, {"host": HOST, "port": 8766})
    assert _policy_rules_allowing(policy, HOST, 8766) == [("tools", [])]
    with pytest.raises(ParseError):
        _resolve_policy(policy, openshell_pb2.SandboxSpec())


def test_the_proto_field_name_network_policies_is_read_too():
    policy = {"network_policies": {"tools": {"endpoints": [{"host": HOST, "port": 8766}]}}}
    assert _policy_rules_allowing(policy, HOST, 8766) == [("tools", [])]


_CAMEL = {"camel": {"endpoints": [{"host": HOST, "port": 8766}]}}
_SNAKE = {"snake": {"endpoints": [{"host": HOST, "port": 8766}]}}


@pytest.mark.parametrize(
    "policy,rules",
    [
        # protobuf's ParseDict, which fills the sandbox's policy, keeps the last of the two names it is given.
        ({"networkPolicies": _CAMEL, "network_policies": _SNAKE}, [("snake", [])]),
        ({"network_policies": _SNAKE, "networkPolicies": _CAMEL}, [("camel", [])]),
        ({"networkPolicies": _CAMEL, "network_policies": None}, []),
        ({"network_policies": None, "networkPolicies": _CAMEL}, [("camel", [])]),
    ],
)
def test_given_both_names_for_the_network_policies_the_last_wins(policy, rules):
    assert _policy_rules_allowing(policy, HOST, 8766) == rules


@pytest.mark.parametrize(
    "policy,camel",
    [
        ({"networkPolicies": _CAMEL, "network_policies": _SNAKE}, False),
        ({"network_policies": _SNAKE, "networkPolicies": _CAMEL}, True),
    ],
)
def test_the_last_name_wins_in_the_policy_the_sandbox_gets_too(policy, camel):
    # The check follows what _resolve_policy (protobuf's ParseDict) does with the installed proto.
    spec = openshell_pb2.SandboxSpec()
    _resolve_policy(policy, spec)
    assert set(MessageToDict(spec.policy)["networkPolicies"]) == ({"camel"} if camel else {"snake"})


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
    with caplog.at_level(logging.WARNING, logger=openshell_runtime.__name__):
        _start(monkeypatch, runtime)
    [warning] = _messages(caplog, "WARNING")
    assert f"allows the host-side interceptor at {HOST}:8766" in warning
    assert "tools" in warning and "/usr/bin/node, /usr/bin/pi" in warning and "isn't verified" in warning


def test_a_rule_restricted_to_binaries_only_warns_even_with_require_policy_allows(monkeypatch, caplog):
    policy = _policy({"host": HOST, "port": 8766}, binaries=["/usr/bin/pi"])
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy, require_policy_allows=True)
    runtime.configure(cluster="test", interceptor_url=INTERCEPTOR_URL)
    with caplog.at_level(logging.WARNING, logger=openshell_runtime.__name__):
        connect = _start(monkeypatch, runtime)
    connect.assert_called_once()
    [warning] = _messages(caplog, "WARNING")
    assert "/usr/bin/pi" in warning


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
    assert _messages(caplog, "WARNING") == []


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
