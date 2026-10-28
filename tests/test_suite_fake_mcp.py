"""The bundled suites' simulated write tools change the environment at the revision they read."""

from typing import Any

import pytest
from midojo_sdk.context import ToolContext

from midojo.control_plane.inprocess import InProcessAgentClient
from midojo.control_plane.main import create_app
from midojo.control_plane.service import BySession
from midojo.suites import get_suite
from suites.minibank.a2a_agent import fake_mcp as minibank
from suites.weather import host_tools as weather


class Interleaved(InProcessAgentClient):
    """An in-process world in which ``concurrent`` lands between the tool's first read and its first change."""

    def __init__(self, control, session_token: str, concurrent: dict[str, Any]) -> None:
        super().__init__(control, session_token)
        self.other = InProcessAgentClient(control, session_token)
        self.concurrent: dict[str, Any] | None = concurrent
        self.changes: list[int | None] = []

    async def change(self, values, *, expected_revision=None):
        if self.concurrent is not None:
            await self.other.change(self.concurrent)
            self.concurrent = None
        changed = await super().change(values, expected_revision=expected_revision)
        self.changes.append(expected_revision if changed is None else changed.revision)
        return changed


def open_evaluation(suite_name: str, user_task_id: str):
    control = create_app({suite_name: get_suite(suite_name)}).state.control
    created = control.create_evaluation(control.create_run(suite_name).id, user_task_id)
    return control, created.session_token


def environment(control, token: str) -> dict[str, Any]:
    return control.evaluation(BySession(token)).environment.model_dump(mode="json")


@pytest.mark.asyncio
async def test_a_weather_alert_keeps_an_alert_added_concurrently():
    control, token = open_evaluation("weather", "weather_new_york")
    storm = {"city": "Boston", "message": "storm"}
    world = Interleaved(control, token, {"weather_alerts": [storm]})

    answer = await weather.send_weather_alert(ToolContext(world), city="Lisbon", message="heat")

    assert answer == "Weather alert sent for Lisbon: heat"
    assert environment(control, token)["weather_alerts"] == [storm, {"city": "Lisbon", "message": "heat"}]
    assert len(world.changes) == 2  # the first change was refused at the stale revision, the second applied


@pytest.mark.asyncio
async def test_a_minibank_notification_keeps_one_sent_concurrently():
    control, token = open_evaluation("minibank", "account_balance")
    other = {"recipient": "ops", "message": "audit", "id": "NOTIF-OTHER"}
    world = Interleaved(control, token, {"notifications": [other]})

    await minibank.send_notification(ToolContext(world), recipient="alice", message="hello")

    notifications = environment(control, token)["notifications"]
    assert [n["recipient"] for n in notifications] == ["ops", "alice"]


@pytest.mark.asyncio
async def test_a_minibank_transfer_debits_the_balance_as_changed_concurrently():
    control, token = open_evaluation("minibank", "account_balance")
    accounts = environment(control, token)["accounts"]
    source, destination = list(accounts)[:2]
    accounts[source]["balance"] = 1000.0
    accounts[destination]["balance"] = 0.0
    await InProcessAgentClient(control, token).change({"accounts": accounts})
    # A concurrent deposit to the source account lands between the tool's read and its change.
    deposited = {**accounts, source: {**accounts[source], "balance": 1500.0}}
    world = Interleaved(control, token, {"accounts": deposited})

    answer = await minibank.initiate_transfer(
        ToolContext(world),
        from_account=source,
        to_account=destination,
        amount=100.0,
        description="rent",
        is_cross_border=False,
    )

    assert '"COMPLETED"' in answer
    balances = {k: v["balance"] for k, v in environment(control, token)["accounts"].items()}
    assert (balances[source], balances[destination]) == (1400.0, 100.0)


@pytest.mark.asyncio
async def test_minibank_flagging_sets_both_fields_in_one_change():
    control, token = open_evaluation("minibank", "account_balance")
    account = next(iter(environment(control, token)["accounts"]))
    world = Interleaved(control, token, {"flagged_accounts": ["ACC-OTHER"]})

    await minibank.flag_suspicious_activity(ToolContext(world), account_id=account, reason="structuring")

    after = environment(control, token)
    assert after["flagged_accounts"] == ["ACC-OTHER", account]
    assert after["accounts"][account]["is_flagged"] is True
    # One refused change at the stale revision, then one change setting both fields.
    assert len(world.changes) == 2
    assert control.evaluation(BySession(token)).environment_revision == world.changes[-1]


@pytest.mark.asyncio
async def test_a_write_tool_gives_up_when_the_environment_keeps_changing():
    control, token = open_evaluation("minibank", "account_balance")

    class Racing(InProcessAgentClient):
        async def change(self, values, *, expected_revision=None):
            await InProcessAgentClient(control, token).change({"audit_log": []})
            return await super().change(values, expected_revision=expected_revision)

    with pytest.raises(RuntimeError, match="kept changing"):
        await minibank.send_notification(ToolContext(Racing(control, token)), recipient="alice", message="hi")
    assert environment(control, token)["notifications"] == []
