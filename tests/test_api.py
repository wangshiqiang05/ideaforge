import time
from pathlib import Path

import pytest
from conftest import requires_loopback
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from ideaforge.api.main import create_app
from ideaforge.api.tasks import (
    ApprovalRequest,
    BudgetDecisionRequest,
    CreateTaskRequest,
    approve,
    budget_decision,
    create_task,
    get_cycle,
    get_report,
    get_task,
)
from ideaforge.errors import AgentFailure
from ideaforge.state import (
    CycleRecord,
    FusionPlan,
    PipelineState,
    TaskStatus,
    ToolReport,
)


@pytest.fixture()
def app(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    return TestClient(create_app(config_path=str(config_path), workspace_root=tmp_path / "ws",
                                 pipeline_factory=make_fake_factory()))


def make_fake_factory():
    def factory(cfg, task_id, journal, root, fsm, hook):
        return FakeOrchestrator(fsm, journal, hook)

    return factory


class FakeOrchestrator:
    """Suspend at the planning approval gate, then complete on approval."""

    def __init__(self, fsm, journal, hook):
        self.fsm = fsm
        self.journal = journal
        self.hook = hook

    def run(self, state: PipelineState) -> PipelineState:
        plan = FusionPlan.model_validate({
            "base_repo": "b",
            "donors": [{"repo": "d", "adopt": [{"path": "a.py"}]}],
            "acceptance": [{"item_id": "A1", "command": "pytest"}],
        })
        self.journal.emit("stage_enter", {"stage": "Planning"})
        self.fsm.transition(TaskStatus.AWAITING_APPROVAL, "plan ready (fake)")
        self.journal.emit("approval_required", {"kind": "plan", "plan_version": 1})
        decision = self.hook({"kind": "plan"})
        if decision.decision == "reject":
            self.fsm.transition(TaskStatus.FAILED, "user rejected fusion plan")
            self.journal.emit("failed", {"reason": "user rejected fusion plan"})
            raise AgentFailure("user rejected fusion plan")
        self.fsm.transition(TaskStatus.RUNNING, "approved")
        self.journal.emit("stage_exit", {"stage": "Planning", "failed": False})
        self.fsm.transition(TaskStatus.COMPLETED, "fake convergence")
        self.journal.emit("completed", {})
        # carry the fsm's final task so API state queries see the terminal status
        return state.evolved(fusion_plan=plan, task=self.fsm.task)


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@requires_loopback
def test_idea_validation_errors(app):
    resp = app.post("/tasks", json={"idea": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "idea_empty"

    resp = app.post("/tasks", json={"idea": "x" * 5001})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "idea_too_long"
    assert detail["actual"] == 5001 and detail["limit"] == 5000


@requires_loopback
def test_invalid_config_reports_items(app):
    resp = app.post("/tasks", json={"idea": "good idea", "config": {"budget_usd": -1}})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_config"
    assert any("budget_usd" in item for item in detail["items"])


@requires_loopback
def test_create_and_get_task(app):
    resp = app.post("/tasks", json={"idea": "build a backtest"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "running"
    task_id = body["task_id"]

    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "awaiting_approval")
    got = app.get(f"/tasks/{task_id}").json()
    assert got["cost"]["total"] == 0.0
    assert got["stage"] == "created"

    # 409 guards while not suspended
    resp = app.post(f"/tasks/{task_id}/budget-decision", json={"decision": "abort"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "not_awaiting_budget_decision"

    resp = app.get(f"/tasks/{task_id}/report")
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "task_not_finished"

    # approve -> pipeline resumes and completes
    resp = app.post(f"/tasks/{task_id}/approval", json={"decision": "approve", "gpl_accepted": False})
    assert resp.status_code == 200
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "completed")

    # second approval now conflicts (no pending decision)
    resp = app.post(f"/tasks/{task_id}/approval", json={"decision": "approve"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "not_awaiting_approval"

    # report now available
    resp = app.get(f"/tasks/{task_id}/report")
    assert resp.status_code == 409  # fake orchestrator wrote no report.json
    assert resp.json()["detail"]["error"] == "task_not_finished"


@requires_loopback
def test_reject_approval_fails_task(app):
    resp = app.post("/tasks", json={"idea": "another idea"})
    task_id = resp.json()["task_id"]
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "awaiting_approval")
    resp = app.post(f"/tasks/{task_id}/approval", json={"decision": "reject"})
    assert resp.status_code == 200
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "failed")


@requires_loopback
def test_unknown_task_404(app):
    assert app.get("/tasks/nope").status_code == 404
    assert app.get("/tasks/nope/cycles/1").status_code == 404
    assert app.post("/tasks/nope/approval", json={"decision": "approve"}).status_code == 404


@requires_loopback
def test_cycle_detail_endpoint(app):
    resp = app.post("/tasks", json={"idea": "cycle view"})
    task_id = resp.json()["task_id"]
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "awaiting_approval")
    app.post(f"/tasks/{task_id}/approval", json={"decision": "approve"})
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "completed")
    # fake orchestrator produced no cycles -> 404
    assert app.get(f"/tasks/{task_id}/cycles/1").status_code == 404


@requires_loopback
def test_sse_stream_events(app):
    resp = app.post("/tasks", json={"idea": "stream me"})
    task_id = resp.json()["task_id"]
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "awaiting_approval")
    app.post(f"/tasks/{task_id}/approval", json={"decision": "approve"})
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "completed")

    with app.stream("GET", f"/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "task_created" in body
    assert "completed" in body
    assert "event_seq" in body


@requires_loopback
def test_sse_resume_from_last_event_id(app):
    resp = app.post("/tasks", json={"idea": "resume stream"})
    task_id = resp.json()["task_id"]
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "awaiting_approval")
    app.post(f"/tasks/{task_id}/approval", json={"decision": "approve"})
    assert wait_until(lambda: app.get(f"/tasks/{task_id}").json().get("status") == "completed")

    with app.stream(
        "GET", f"/tasks/{task_id}/events", headers={"Last-Event-ID": "2"}
    ) as response:
        body = "".join(response.iter_text())
    parsed_seq = [int(line.split("id: ")[1].split()[0]) for line in body.splitlines()
                  if line.startswith("id: ")]
    assert parsed_seq and min(parsed_seq) >= 3  # gap replay: nothing before seq 2

# --------------------------------------------------------------------------
# V1-V6: function-level substitute verification for the loopback-skipped
# endpoint tests above (review_fix design 2.1.3(3)). Route functions in
# ideaforge/api/tasks.py are plain sync functions; the socket dependence of
# the skipped tests lives solely in the HTTP transport layer, so the same
# business assertions run here against the real create_app assembly plus a
# minimal Request double (no socket, no asyncio).
# --------------------------------------------------------------------------


def _function_level_app(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    return create_app(
        config_path=str(config_path),
        workspace_root=tmp_path / "ws",
        pipeline_factory=make_fake_factory(),
    )


def _make_request(app):
    return Request(scope={"type": "http", "app": app})


def _wait_awaiting_approval(app, task_id):
    assert wait_until(
        lambda: app.state.registry.get(task_id).state_ref.task.status.value == "awaiting_approval"
    )
    assert wait_until(lambda: app.state.registry.get(task_id).pending is not None)


def test_v1_idea_validation_function_level(tmp_path):
    """V1 <- test_idea_validation_errors (skipped): 400 idea_empty / idea_too_long."""
    req = _make_request(_function_level_app(tmp_path))

    with pytest.raises(HTTPException) as exc:
        create_task(CreateTaskRequest(idea="   "), req)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "idea_empty"

    with pytest.raises(HTTPException) as exc:
        create_task(CreateTaskRequest(idea="x" * 5001), req)
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert detail["error"] == "idea_too_long"
    assert detail["actual"] == 5001 and detail["limit"] == 5000


def test_v2_invalid_config_function_level(tmp_path):
    """V2 <- test_invalid_config_reports_items (skipped): 400 invalid_config items."""
    req = _make_request(_function_level_app(tmp_path))
    with pytest.raises(HTTPException) as exc:
        create_task(CreateTaskRequest(idea="good idea", config={"budget_usd": -1}), req)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "invalid_config"
    assert any("budget_usd" in item for item in exc.value.detail["items"])


def test_v3_create_get_and_conflict_guards(tmp_path):
    """V3 <- test_create_and_get_task (skipped): 202 create, cost/stage, 409 guards."""
    app = _function_level_app(tmp_path)
    req = _make_request(app)
    body = create_task(CreateTaskRequest(idea="build a backtest"), req)
    assert body["status"] == "running"
    task_id = body["task_id"]

    _wait_awaiting_approval(app, task_id)
    got = get_task(task_id, req)
    assert got["cost"]["total"] == 0.0
    assert got["stage"] == "created"

    # 409 guards while not suspended
    with pytest.raises(HTTPException) as exc:
        budget_decision(task_id, BudgetDecisionRequest(decision="abort"), req)
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "not_awaiting_budget_decision"

    with pytest.raises(HTTPException) as exc:
        get_report(task_id, req)
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "task_not_finished"

    # approve -> pipeline resumes and completes
    assert approve(task_id, ApprovalRequest(decision="approve", gpl_accepted=False), req) == {
        "status": "received"
    }
    assert wait_until(lambda: get_task(task_id, req)["status"] == "completed")

    # second approval now conflicts (no pending decision)
    with pytest.raises(HTTPException) as exc:
        approve(task_id, ApprovalRequest(decision="approve"), req)
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "not_awaiting_approval"

    # fake orchestrator wrote no report.json
    with pytest.raises(HTTPException) as exc:
        get_report(task_id, req)
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "task_not_finished"


def test_v4_reject_approval_fails_task(tmp_path):
    """V4 <- test_reject_approval_fails_task (skipped): reject -> failed."""
    app = _function_level_app(tmp_path)
    req = _make_request(app)
    body = create_task(CreateTaskRequest(idea="another idea"), req)
    task_id = body["task_id"]
    _wait_awaiting_approval(app, task_id)
    assert approve(task_id, ApprovalRequest(decision="reject"), req) == {"status": "received"}
    assert wait_until(lambda: get_task(task_id, req)["status"] == "failed")


def test_v5_unknown_task_404_function_level(tmp_path):
    """V5 <- test_unknown_task_404 (skipped): 404 on three endpoints."""
    req = _make_request(_function_level_app(tmp_path))
    with pytest.raises(HTTPException) as exc:
        get_task("nope", req)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        get_cycle("nope", 1, req)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        approve("nope", ApprovalRequest(decision="approve"), req)
    assert exc.value.status_code == 404


def test_v6_cycle_detail_function_level(tmp_path):
    """V6 <- test_cycle_detail_endpoint (skipped): cycle detail 404 + 200 shape."""
    app = _function_level_app(tmp_path)
    req = _make_request(app)
    body = create_task(CreateTaskRequest(idea="cycle view"), req)
    task_id = body["task_id"]
    _wait_awaiting_approval(app, task_id)
    approve(task_id, ApprovalRequest(decision="approve"), req)
    assert wait_until(lambda: get_task(task_id, req)["status"] == "completed")

    # fake orchestrator produced no cycles -> 404 (same assertion as skipped test)
    with pytest.raises(HTTPException) as exc:
        get_cycle(task_id, 1, req)
    assert exc.value.status_code == 404

    # 200 branch: pre-seeded CycleRecord returns the documented payload shape
    rt = app.state.registry.get(task_id)
    rt.state_ref = rt.state_ref.model_copy(
        update={"cycles": [CycleRecord(cycle=1, tool_report=ToolReport(passed=True))]}
    )
    detail = get_cycle(task_id, 1, req)
    assert detail["cycle"] == 1
    assert detail["tool_report"]["passed"] is True