import pytest
from fastapi.testclient import TestClient

from deep_research import api
from deep_research.graph import Researcher
from tests.conftest import TOPIC, scripted

GOOD = {"score": 9, "missing": [], "unsupported": [], "follow_up_queries": []}


@pytest.fixture
def client(settings, search, monkeypatch):
    import dataclasses

    instances = {flag: Researcher(dataclasses.replace(settings, approve_plan=flag), llm=scripted([GOOD]),
                                  search=search) for flag in (False, True)}
    monkeypatch.setattr(api, "researcher", lambda approve_plan=False: instances[approve_plan])
    api._errors.clear()
    return TestClient(api.app)


def test_start_and_poll(client):
    r = client.post("/research", json={"topic": TOPIC})
    assert r.status_code == 202
    thread = r.json()["thread_id"]
    body = client.get(f"/research/{thread}").json()  # TestClient runs background tasks before returning
    assert body["status"] == "done" and "## References" in body["report"]


def test_plan_review_flow(client):
    thread = client.post("/research", json={"topic": TOPIC, "approve_plan": True}).json()["thread_id"]
    assert client.get(f"/research/{thread}").json()["status"] == "awaiting_approval"
    assert client.post(f"/research/{thread}/approve", json={"approved": True}).status_code == 202
    assert client.get(f"/research/{thread}").json()["status"] == "done"
    assert client.post(f"/research/{thread}/approve", json={"approved": True}).status_code == 409


def test_unknown_thread_and_validation(client):
    assert client.get("/research/nope").status_code == 404
    assert client.post("/research", json={"topic": "hi"}).status_code == 422
