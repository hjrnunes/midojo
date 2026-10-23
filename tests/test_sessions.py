"""Exercise callback isolation across suites, app instances, and overlapping tasks."""

import asyncio
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx
import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from midojo_sdk.context import ToolContext
from midojo_sdk.recorder import ControlPlaneRejectedError, InvalidRequestError, RejectionKind
from midojo_sdk.session import MidojoSessionMiddleware, MissingSessionError, session_context, session_token
from pydantic import ValidationError

from midojo.control_plane.config import AppConfig
from midojo.control_plane.main import create_app
from midojo.control_plane.service import ById
from midojo.yaml_task_suite import YAMLTaskSuite
from tests.backends import Agent


def make_suite(tmp_path, name, field, value):
    path = tmp_path / f"{name}.yaml"
    path.write_text(f"""
agent_runtime: {{type: unmanaged}}
environment:
  {field}: {value}
user_tasks:
  - id: shared_task
    prompt: Work on {name}
    utility:
      env_field_equals:
        field: {field}
        value: {value}
""")
    return YAMLTaskSuite(name, path)


def new_evaluation(client, suite_name="weather", task_id="weather_new_york"):
    response = client.post("/runs", json={"suite_name": suite_name})
    assert response.status_code == 201, response.text
    run = response.json()
    response = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": task_id})
    assert response.status_code == 201, response.text
    return run, response.json()


def auth(evaluation):
    return {"Authorization": f"Bearer {evaluation['session_token']}"}


@pytest.mark.asyncio
async def test_interleaved_suites_validate_and_grade_their_own_environment(tmp_path, backend):
    first = make_suite(tmp_path, "first", "count", 3)
    second = make_suite(tmp_path, "second", "label", "ready")
    app = create_app({"first": first, "second": second})
    client = TestClient(app)
    run_a, a = new_evaluation(client, "first", "shared_task")
    run_b, b = new_evaluation(client, "second", "shared_task")
    agent_a, agent_b = backend.agent(a["session_token"], app), backend.agent(b["session_token"], app)
    assert client.get("/suites").json() == ["first", "second"]
    assert await agent_a.read() == (0, {"count": 3})
    assert await agent_b.read() == (0, {"label": "ready"})
    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent_a.change({"label": "wrong"})
    assert refused.value.kind is RejectionKind.INVALID_ENVIRONMENT
    assert await agent_a.read() == (0, {"count": 3})  # neither the environment nor its revision changed
    assert await agent_b.change({"label": "changed"}) == (1, {"label": "changed"})
    for run, ev, agent, expected in [(run_a, a, agent_a, True), (run_b, b, agent_b, False)]:
        url = f"/runs/{run['id']}/evaluations/{ev['id']}"
        client.post(f"{url}/complete", json={"agent_output": "done"})
        assert client.post(f"{url}/grade").json()["utility"] is expected
        assert client.get(f"/runs/{run['id']}").json()["suite_name"] == run["suite_name"]
        with pytest.raises(ControlPlaneRejectedError) as closed:
            await agent.read()
        assert closed.value.kind is RejectionKind.SESSION_CLOSED


def test_an_environment_put_is_validated_against_its_own_suite(tmp_path):
    first = make_suite(tmp_path, "first", "count", 3)
    second = make_suite(tmp_path, "second", "label", "ready")
    client = TestClient(create_app({"first": first, "second": second}))
    _, a = new_evaluation(client, "first", "shared_task")
    _, b = new_evaluation(client, "second", "shared_task")
    assert client.put("/agent/environment", headers=auth(a), json={"label": "wrong"}).status_code == 422
    assert client.put("/agent/environment", headers=auth(b), json={"label": "changed"}).status_code == 200


@pytest.mark.asyncio
async def test_parallel_callbacks_stay_with_their_session(client, backend):
    run_a, a = new_evaluation(client)
    run_b, b = new_evaluation(client)

    async def report(i):
        ev = a if i % 2 else b
        await backend.agent(ev["session_token"]).finish_call(
            call_id=str(uuid.uuid4()), function="report", args={}, result=ev["id"], disposition="simulated"
        )

    await asyncio.gather(*(report(i) for i in range(32)))
    for run, ev in [(run_a, a), (run_b, b)]:
        calls = client.get(f"/runs/{run['id']}/evaluations/{ev['id']}/function-calls").json()
        assert len(calls) == 16
        assert {call["result"] for call in calls} == {ev["id"]}


def test_parallel_http_callbacks_stay_with_their_session(app):
    # The context-managed client sends every request to one event loop, like uvicorn.
    with TestClient(app) as client:
        run_a, a = new_evaluation(client)
        run_b, b = new_evaluation(client)

        def report(i):
            ev = a if i % 2 else b
            response = client.post(
                "/agent/function-calls",
                headers=auth(ev),
                json={
                    "function": "report",
                    "args": {},
                    "result": ev["id"],
                },
            )
            assert response.status_code == 201

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(report, range(32)))
        for run, ev in [(run_a, a), (run_b, b)]:
            calls = client.get(f"/runs/{run['id']}/evaluations/{ev['id']}/function-calls").json()
            assert len(calls) == 16
            assert {call["result"] for call in calls} == {ev["id"]}


def test_sessions_are_private_over_http(client):
    run, ev = new_evaluation(client)
    url = f"/runs/{run['id']}/evaluations/{ev['id']}"
    assert "session_token" not in client.get(url).json()
    assert ev["session_token"] not in client.get(f"/runs/{run['id']}").text
    assert client.get("/agent/environment").status_code == 401
    assert client.get("/agent/environment", headers={"Authorization": "Bearer unknown"}).status_code == 401


@pytest.mark.asyncio
async def test_sessions_expire_and_cannot_fall_back(suite, monkeypatch, backend):
    now = [1_700_000_000]
    monkeypatch.setattr("midojo.control_plane.service.time.time", lambda: now[0])
    app = create_app({"weather": suite}, config=AppConfig(session_ttl_seconds=5))
    client = TestClient(app)
    run, ev = new_evaluation(client)
    url = f"/runs/{run['id']}/evaluations/{ev['id']}"
    agent = backend.agent(ev["session_token"], app)
    with pytest.raises(ControlPlaneRejectedError) as unknown:
        await backend.agent("unknown", app).read()
    assert unknown.value.kind is RejectionKind.INVALID_SESSION
    now[0] += 4
    assert (await agent.read()).revision == 0
    now[0] += 1
    for write in [
        agent.read(),
        agent.finish_call(call_id="late", function="late", args={}, result="late", disposition="simulated"),
        agent.start_call(call_id="late", function="late", args={}, disposition="simulated"),
        agent.change({"weather_alerts": []}),
    ]:
        with pytest.raises(ControlPlaneRejectedError) as expired:
            await write
        assert expired.value.kind is RejectionKind.INVALID_SESSION
    assert client.get(f"{url}/function-calls").json() == []


@pytest.mark.asyncio
async def test_explicit_close_does_not_affect_another_session(client, backend):
    run, ev = new_evaluation(client)
    _, other = new_evaluation(client)
    url = f"/runs/{run['id']}/evaluations/{ev['id']}/session"
    assert client.delete(url).status_code == 204
    assert client.delete(url).status_code == 204
    with pytest.raises(ControlPlaneRejectedError) as closed:
        await backend.agent(ev["session_token"]).read()
    assert closed.value.kind is RejectionKind.INVALID_SESSION
    assert (await backend.agent(other["session_token"]).read()).revision == 0


@pytest.mark.asyncio
async def test_apps_have_independent_suites_stores_and_routes(suite, backend):
    suites = {"first": suite}
    first_app, second_app = create_app(suites), create_app({"second": suite})
    first, second = TestClient(first_app), TestClient(second_app)
    suites.clear()
    run, ev = new_evaluation(first, "first")
    assert first.get("/suites").json() == ["first"]
    assert second.get("/suites").json() == ["second"]
    assert second.get(f"/runs/{run['id']}").status_code == 404
    with pytest.raises(ControlPlaneRejectedError) as foreign:
        await backend.agent(ev["session_token"], second_app).read()
    assert foreign.value.kind is RejectionKind.INVALID_SESSION
    assert (await backend.agent(ev["session_token"], first_app).read()).revision == 0


@pytest.mark.asyncio
async def test_concurrent_environment_changes_through_tools_are_not_lost(client, backend):
    _, ev = new_evaluation(client)
    agent = backend.agent(ev["session_token"])
    cities = (await agent.read()).environment["cities"]
    sunny = {name: {**city, "condition": "sunny"} for name, city in cities.items()}
    alerts = [{"city": "Boston", "message": "storm"}]

    await asyncio.gather(
        *(ToolContext(backend.agent(ev["session_token"])).env_update("weather_alerts", alerts) for _ in range(4)),
        *(ToolContext(backend.agent(ev["session_token"])).env_update("cities", sunny) for _ in range(4)),
    )

    assert await agent.read() == (8, {"cities": sunny, "weather_alerts": alerts})


@pytest.mark.asyncio
async def test_a_change_at_a_stale_revision_changes_nothing(client, backend):
    _, ev = new_evaluation(client)
    agent = backend.agent(ev["session_token"])
    ctx = ToolContext(agent)
    revision, environment = await ctx.read()
    assert await ctx.env_update("weather_alerts", [{"city": "Boston", "message": "first"}], expected_revision=revision)

    stale = [{"city": "Boston", "message": "stale"}]
    assert await ctx.env_update("weather_alerts", stale, expected_revision=revision) is False
    assert await agent.change({"weather_alerts": stale}, expected_revision=revision) is None

    assert await agent.read() == (
        revision + 1,
        {**environment, "weather_alerts": [{"city": "Boston", "message": "first"}]},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "expected_revision"),
    [
        ({"weather_alerts": []}, "bad"),
        ({"weather_alerts": []}, 1.5),
        ([{"weather_alerts": []}], None),
        ("weather_alerts", None),
    ],
    ids=["revision-not-an-integer", "revision-fractional", "values-a-list", "values-a-string"],
)
async def test_a_malformed_change_is_an_invalid_request_on_both_implementations(
    client, backend, values, expected_revision
):
    _, ev = new_evaluation(client)
    agent = backend.agent(ev["session_token"])
    before = await agent.read()

    with pytest.raises(InvalidRequestError):
        await agent.change(values, expected_revision=expected_revision)  # pyright: ignore[reportArgumentType]

    assert await agent.read() == before


@pytest.mark.asyncio
async def test_a_coercible_revision_means_the_same_on_both_implementations(client, backend):
    """The request model coerces "0" to 0 over HTTP, and so does the in-process client."""
    _, ev = new_evaluation(client)
    agent = backend.agent(ev["session_token"])
    _, environment = await agent.read()
    alerts = [{"city": "Boston", "message": "storm"}]

    changed = await agent.change({"weather_alerts": alerts}, expected_revision="0")  # pyright: ignore[reportArgumentType]
    stale = await agent.change({"weather_alerts": []}, expected_revision="0")  # pyright: ignore[reportArgumentType]

    assert changed == (1, {**environment, "weather_alerts": alerts})
    assert stale is None
    assert await agent.read() == (1, {**environment, "weather_alerts": alerts})


@pytest.mark.asyncio
async def test_an_environment_change_after_the_seal_is_the_same_late_fact(client, backend):
    run, ev = new_evaluation(client)
    url = f"/runs/{run['id']}/evaluations/{ev['id']}"
    agent = backend.agent(ev["session_token"])
    assert await agent.change({"weather_alerts": [{"city": "Boston", "message": "before"}]}) is not None
    client.post(f"{url}/complete", json={"agent_output": "done"})
    sealed = client.get(url).json()

    late = [{"city": "Boston", "message": "late"}]
    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent.change({"weather_alerts": late}, expected_revision=1)

    assert (refused.value.kind, refused.value.late_fact) == (RejectionKind.SESSION_CLOSED, True)
    after = client.get(url).json()
    assert after["environment"] == sealed["environment"]
    assert backend.app.state.control.evaluation(ById(run["id"], ev["id"])).environment_revision == 1
    [fact] = after["late_facts"]
    del fact["at"]
    assert fact == {
        "event": "change_environment",
        "stage": "sealed",
        "by": "agent",
        "evidence": ["environment"],
        "payload": {"set": {"weather_alerts": late}, "expected_revision": 1},
        "call_id": None,
    }


@pytest.mark.asyncio
async def test_a_tool_context_without_a_world_has_no_environment():
    with pytest.raises(RuntimeError, match="no WorldState"):
        await ToolContext().env("cities")


def test_run_requires_loaded_suite(client):
    assert client.post("/runs").status_code == 422
    response = client.post("/runs", json={"suite_name": "weather"})
    assert response.status_code == 201


def test_suite_name_constraint_applies_to_app_requests_and_paths(client, suite):
    name = "bad name"
    # Registered aliases must be checked even when the suite object's own name is valid.
    with pytest.raises(ValidationError):
        create_app({name: suite})
    response = client.post("/runs", json={"suite_name": name})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "suite_name"]
    response = client.get(f"/suites/{quote(name, safe='')}/tasks/user")
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "suite_name"]


@pytest.mark.asyncio
async def test_remote_mcp_server_reads_session_from_each_request(app, client):
    from midojo_sdk.mcp import MidojoMCP, ToolContext

    run_a, a = new_evaluation(client)
    run_b, b = new_evaluation(client)
    control_http = httpx.AsyncClient(transport=httpx.ASGITransport(app))
    mcp = MidojoMCP("report", control_plane_url="http://control", http=control_http)

    @mcp.tool()
    async def report(ctx: ToolContext, message: str) -> str:
        assert "New York" in await ctx.env("cities")
        await ctx.env_update("weather_alerts", [{"city": "New York", "message": message}])
        return message

    mcp_app = mcp._fastmcp.http_app(path="/mcp", stateless_http=True)
    async with mcp_app.router.lifespan_context(mcp_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(mcp_app), base_url="http://mcp") as http:
            responses = await asyncio.gather(
                *[
                    http.post(
                        "/mcp",
                        headers={
                            "X-Midojo-Session": ev["session_token"],
                            "Accept": "application/json, text/event-stream",
                        },
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": "report", "arguments": {"message": ev["id"]}},
                        },
                    )
                    for ev in [a, b]
                ]
            )
            assert all(response.status_code == 200 for response in responses)
    for run, ev in [(run_a, a), (run_b, b)]:
        calls = client.get(f"/runs/{run['id']}/evaluations/{ev['id']}/function-calls").json()
        assert [call["result"] for call in calls] == [ev["id"]]
        environment = client.get(f"/runs/{run['id']}/evaluations/{ev['id']}/environment").json()
        assert environment["weather_alerts"] == [{"city": "New York", "message": ev["id"]}]
    await control_http.aclose()


@pytest.mark.asyncio
async def test_a2a_transport_keeps_executor_callbacks_scoped(app, client, monkeypatch, backend):
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes.agent_card_routes import create_agent_card_routes
    from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, Message, Part, Role
    from starlette.applications import Starlette

    from midojo.agent_client import A2AAgentClient

    monkeypatch.delenv("MIDOJO_SESSION_TOKEN", raising=False)
    run_a, a = new_evaluation(client)
    run_b, b = new_evaluation(client)
    backend.http()  # opened before httpx.AsyncClient is patched below

    def sdk() -> Agent:
        return backend.agent(session_token())

    class Executor(AgentExecutor):
        async def execute(self, context, event_queue):
            assert context.message is not None
            prompt = context.message.parts[0].text
            await asyncio.sleep(0)
            await sdk().finish_call(
                call_id=str(uuid.uuid4()), function="a2a_task", args={}, result=prompt, disposition="simulated"
            )
            await event_queue.enqueue_event(Message(role=Role.ROLE_AGENT, parts=[Part(text=prompt)]))

        async def cancel(self, context, event_queue):
            raise NotImplementedError

    card = AgentCard(
        name="test",
        description="test",
        version="1",
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(url="http://agent/", protocol_binding="JSONRPC")],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=InMemoryTaskStore(), agent_card=card)
    agent = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])
    agent.add_middleware(MidojoSessionMiddleware)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(**kw, transport=httpx.ASGITransport(agent)))
    results = await asyncio.gather(
        *[A2AAgentClient("http://agent/").send_task(ev["id"], session_token=ev["session_token"]) for ev in [a, b]]
    )
    assert results == [a["id"], b["id"]]
    for run, ev in [(run_a, a), (run_b, b)]:
        calls = client.get(f"/runs/{run['id']}/evaluations/{ev['id']}/function-calls").json()
        assert [call["result"] for call in calls] == [ev["id"]]
    with pytest.raises(MissingSessionError):
        await sdk().read()
    client.delete(f"/runs/{run_a['id']}/evaluations/{a['id']}/session")
    with session_context(a["session_token"]), pytest.raises(ControlPlaneRejectedError) as failure:
        await sdk().finish_call(call_id="late", function="late", args={}, result="late", disposition="simulated")
    assert failure.value.kind is RejectionKind.INVALID_SESSION


def test_handlers_and_dependencies_run_on_the_event_loop(app):
    # FastAPI runs sync callables on a thread pool. The writer keeps each
    # evaluation's check-then-write atomic there too, but async handlers keep
    # its lock waits off the pool (see midojo.control_plane.service).
    def dependants(dependant: Dependant) -> Iterator[Dependant]:
        yield dependant
        for sub in dependant.dependencies:
            yield from dependants(sub)

    sync = sorted(
        f"{route.path}: {getattr(dep.call, '__name__', type(dep.call).__name__)}"
        for route in app.routes
        if isinstance(route, APIRoute)
        for dep in dependants(route.dependant)
        if not (dep.is_coroutine_callable or dep.is_async_gen_callable)
    )
    assert sync == [], "declare these async def (see midojo.control_plane.service)"
