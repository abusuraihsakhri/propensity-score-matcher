"""HTTP checks for the optional FastAPI server and static Python browser assets."""
from fastapi.testclient import TestClient
from agents.api import app


def test_browser_root_serves_real_matching_ui():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Propensity Score Matcher" in page.text
    assert "app.js" in page.text
    assert client.get("/app.js").status_code == 200
    module = client.get("/ps_matcher.py")
    assert module.status_code == 200
    assert "def propensity_score_match(" in module.text


def test_audit_sensitive_identifier_is_rejected_without_echo():
    client = TestClient(app)
    response = client.post("/api/audit", json={
        "task_id": "TASK-001",
        "target_identifier": "MRN-994827",
        "primary_metric": 1.0,
    })
    assert response.status_code == 400
    assert "994827" not in response.text


def test_health():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
