def test_task_detail_unknown_user(client):
    resp = client.get("/suites/weather/tasks/user/nonexistent")
    assert resp.status_code == 404


def test_task_detail_unknown_injection(client):
    resp = client.get("/suites/weather/tasks/injection/nonexistent")
    assert resp.status_code == 404


def test_user_task_detail_lists_its_turns(client, suite, monkeypatch):
    assert client.get("/suites/weather/tasks/user/warmest_city").json()["turns"] == [
        "Which city currently has the warmest weather?"
    ]
    monkeypatch.setattr(suite.user_tasks["warmest_city"], "turns", ["Warmest city?", "And the coldest?"])
    detail = client.get("/suites/weather/tasks/user/warmest_city").json()
    assert (detail["prompt"], detail["turns"]) == ("Warmest city?", ["Warmest city?", "And the coldest?"])
