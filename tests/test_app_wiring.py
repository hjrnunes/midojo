"""An app's wiring and auth settings are fixed once ``create_app`` builds it.

Nothing a caller holds can take the control token away from a served
control plane, swap its store or suites, or make it unserved: the settings
are a frozen ``AppConfig``, the control plane's attributes are read-only,
and the app's state entries can't be set, replaced or deleted. Every route
reads the settings and suites through the control plane, so the token a
route requires is the one the control plane opens host mode under.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from midojo.control_plane.config import AppConfig
from midojo.control_plane.lifecycle import RejectedError, Rejection
from midojo.control_plane.main import create_app
from midojo.control_plane.store import InMemoryStore

TOKEN = "control-secret"


def test_the_settings_are_frozen():
    config = AppConfig(control_token=TOKEN)
    with pytest.raises(ValidationError):
        config.control_token = None  # type: ignore[misc]
    assert config.control_token == TOKEN


@pytest.mark.parametrize("entry", ["control", "config", "store", "suites", "added"])
def test_the_apps_state_entries_cant_be_set_replaced_or_deleted(suite, entry):
    app = create_app({"weather": suite}, config=AppConfig(control_token=TOKEN))
    with pytest.raises(AttributeError, match="fixed when the app is built"):
        setattr(app.state, entry, None)
    if entry != "added":
        with pytest.raises(AttributeError, match="fixed when the app is built"):
            delattr(app.state, entry)
    assert app.state.config.control_token == TOKEN


@pytest.mark.parametrize("attribute", ["config", "store", "suites", "served"])
def test_the_control_planes_wiring_is_read_only(suite, attribute):
    app = create_app({"weather": suite}, config=AppConfig(control_token=TOKEN))
    control = app.state.control
    replacement = {"config": AppConfig(), "store": InMemoryStore(), "suites": {}, "served": False}[attribute]
    with pytest.raises(AttributeError):
        setattr(control, attribute, replacement)
    assert control.served is True and control.config.control_token == TOKEN


def test_the_suites_are_read_only_and_the_control_planes_own(suite):
    given = {"weather": suite}
    app = create_app(given)
    control = app.state.control
    # The routes read the control plane's suites: a change to the mapping given doesn't reach them, and they can't
    # be changed in place.
    given.clear()
    assert list(control.suites) == ["weather"] and app.state.suites is control.suites
    with pytest.raises(TypeError):
        control.suites["other"] = suite  # type: ignore[index]
    assert TestClient(app).get("/suites").json() == ["weather"]


def test_the_app_state_is_the_control_planes_wiring(suite):
    store = InMemoryStore()
    config = AppConfig(control_token=TOKEN)
    app = create_app({"weather": suite}, store=store, config=config)
    control = app.state.control
    assert (app.state.store, app.state.config) == (control.store, control.config) == (store, config)


def test_a_tokenless_app_stays_tokenless_and_refuses_host_mode(suite):
    app = create_app({"weather": suite})
    control = app.state.control
    with pytest.raises(AttributeError):
        app.state.config = AppConfig(control_token=TOKEN)
    with pytest.raises(ValidationError):
        control.config.control_token = TOKEN  # type: ignore[misc]
    run = control.create_run("weather")
    with pytest.raises(RejectedError) as refused:
        control.create_evaluation(run.id, "weather_new_york", interception="host")
    assert refused.value.kind is Rejection.CONTROL_TOKEN_REQUIRED
    # The routes still take a request without a credential: the open app is what it was built as.
    assert TestClient(app).get("/suites").status_code == 200
