import json

import pytest
from mocks import ExecResult, FakeSandboxManager, make_config, make_journal, prospect

from ideaforge.cost import CostLedger
from ideaforge.delivery import (
    DEGRADED_BANNER,
    AcceptanceRecheckFailure,
    Delivery,
    assert_no_completion_phrasing,
    build_report,
    notice_md,
    rerun_acceptance,
    write_artifacts,
)
from ideaforge.pipeline import PipelineOrchestrator
from ideaforge.state import (
    FusionPlan,
    LicenseKind,
    PipelineState,
    RepoMap,
    Task,
    TaskConfig,
    TaskStateMachine,
    TaskStatus,
)
from ideaforge.workspace import WorkspaceManager


def make_state(tmp_path, status=TaskStatus.RUNNING, with_cycles=True) -> PipelineState:
    task = Task(task_id="t1", idea="idea", config=TaskConfig(max_cycles=6))
    task.status = status
    plan = FusionPlan.model_validate({
        "base_repo": "user/mit",
        "donors": [
            {"repo": "user/gpl", "adopt": [{"path": "util.py"}]},
            {"repo": "user/unk", "adopt": [{"path": "other.py"}]},
        ],
        "acceptance": [
            {"item_id": "A1", "command": "python -c 'print(1)'"},
            {"item_id": "A2", "command": "pytest -q"},
        ],
    })
    repo_maps = [
        RepoMap(repo_full_name="user/mit", summary="m", license_declared=LicenseKind.MIT),
        RepoMap(repo_full_name="user/gpl", summary="g", license_declared=LicenseKind.GPL_3),
        RepoMap(repo_full_name="user/unk", summary="u", license_declared=LicenseKind.UNKNOWN),
    ]
    state = PipelineState(task=task, fusion_plan=plan, repo_maps=repo_maps,
                          prospects=[prospect("user/mit"), prospect("user/gpl")])
    return state


def test_rerun_acceptance_all_pass(tmp_path):
    journal = make_journal(tmp_path, "t1")
    sandbox = FakeSandboxManager()  # default exec -> success
    state = make_state(tmp_path)
    (tmp_path / "ws" / "t1" / "forge").mkdir(parents=True)
    evidence = rerun_acceptance(state, sandbox, journal, make_config())
    assert len(evidence) == 2
    assert all(e["passed"] for e in evidence)
    events, _ = journal.read_events()
    assert [e for e in events if e.type == "acceptance_result"]


def test_rerun_failure_produces_failed_evidence(tmp_path):
    journal = make_journal(tmp_path, "t1")
    sandbox = FakeSandboxManager(exec_script=[
        ExecResult(success=True, exit_code=0, output="ok"),
        ExecResult(success=False, exit_code=1, output="1 failed"),
    ])
    state = make_state(tmp_path)
    (tmp_path / "ws" / "t1" / "forge").mkdir(parents=True)
    evidence = rerun_acceptance(state, sandbox, journal, make_config())
    assert evidence[0]["passed"] is True
    assert evidence[1]["passed"] is False


def test_delivery_stage_routes_back_to_loop_on_failure(tmp_path):
    journal = make_journal(tmp_path, "t1")
    sandbox = FakeSandboxManager(exec_script=[
        ExecResult(success=False, exit_code=1, output="1 failed"),
    ])
    state = make_state(tmp_path)
    (tmp_path / "ws" / "t1" / "forge").mkdir(parents=True)
    delivery = Delivery(sandbox, journal, make_config())
    with pytest.raises(AcceptanceRecheckFailure):
        delivery.run(state)


def test_notice_md_lists_licenses_and_flags_copyleft(tmp_path):
    state = make_state(tmp_path)
    notice = notice_md(state)
    assert "base: user/mit (License: MIT)" in notice
    assert "donor: user/gpl" in notice
    assert "COPYLEFT" in notice and "infect clause" in notice
    assert "donor: user/unk" in notice
    assert "manual license verification required" in notice


def test_completed_report_shape(tmp_path):
    state = make_state(tmp_path)
    state.task.status = TaskStatus.COMPLETED
    evidence = [{"item_id": "A1", "command": "x", "passed": True, "output": ""},
                {"item_id": "A2", "command": "y", "passed": True, "output": ""}]
    report = build_report(state, evidence)
    assert report["completed"] is True
    assert report["acceptance_done"] == 2
    assert report["acceptance_total"] == 2
    assert "banner" not in report


def test_degraded_report_never_claims_completion(tmp_path):
    state = make_state(tmp_path)
    state.task.status = TaskStatus.DEGRADED
    evidence = [{"item_id": "A1", "command": "x", "passed": True, "output": ""},
                {"item_id": "A2", "command": "y", "passed": False, "output": "1 failed"}]
    report = build_report(state, evidence)
    assert report["completed"] is False
    assert report["banner"] == DEGRADED_BANNER
    assert report["acceptance_done"] == 1
    assert report["acceptance_total"] == 2
    assert_no_completion_phrasing(report)  # must not raise


def test_failed_report_shape(tmp_path):
    state = make_state(tmp_path)
    state.task.status = TaskStatus.FAILED
    report = build_report(state, [])
    assert report["completed"] is False


def test_write_artifacts(tmp_path):
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    state = make_state(tmp_path)
    state.task.status = TaskStatus.COMPLETED
    report = build_report(state, [])
    paths = write_artifacts(state, report, workspace)
    assert paths["report"].exists()
    assert paths["notice"].exists()
    assert "user/gpl" in paths["notice"].read_text(encoding="utf-8")


def test_orchestrator_recheck_failure_degrades_when_exhausted(tmp_path):
    """Acceptance re-run failed with cycles exhausted -> honest degraded."""
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    (workspace.forge_dir("t1")).mkdir(parents=True, exist_ok=True)
    sandbox = FakeSandboxManager(exec_script=[ExecResult(success=False, exit_code=1, output="bad")])
    cfg = make_config()
    state = make_state(tmp_path)
    state.task.cycle = 6  # rounds exhausted
    fsm = TaskStateMachine(state.task, journal)

    delivery = Delivery(sandbox, journal, cfg)
    ledger = CostLedger(budget_usd=100.0)
    orch = PipelineOrchestrator(
        None, None, None, None, None, None, delivery, workspace, sandbox,
        journal, fsm, ledger, cfg,
    )
    with pytest.raises(AcceptanceRecheckFailure):
        delivery.run(state)
    out = orch._handle_recheck_failure(state, [])
    assert out.task.status == TaskStatus.DEGRADED
    report = json.loads((tmp_path / "ws" / "t1" / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] is False