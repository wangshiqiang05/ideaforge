import json

import pytest
from mocks import make_journal, make_pool
from pydantic import ValidationError

from ideaforge.agents.architect import (
    ApprovalDecision,
    ApprovalGate,
    Architect,
    validate_fusion_plan,
)
from ideaforge.errors import AgentFailure, StructuredOutputError
from ideaforge.state import (
    LicenseKind,
    PipelineState,
    RepoMap,
    Task,
    TaskConfig,
    TaskStateMachine,
    TaskStatus,
)

GOOD_PLAN = {
    "base_repo": "user/mit",
    "donors": [
        {"repo": "user/gpl", "adopt": [{"path": "pkg/util.py", "symbol": None}],
         "drop": ["docs"], "conflicts": []},
    ],
    "bridge_design": [],
    "acceptance": [{"item_id": "A1", "command": "pytest -q", "description": "smoke"}],
    "rename_map": {"pkg": "forgelib"},
    "license_strategy": {"accepted_copyleft": True, "notice": "user accepted"},
}


def repo_maps():
    return [
        RepoMap(repo_full_name="user/mit", summary="mit repo",
                license_declared=LicenseKind.MIT),
        RepoMap(repo_full_name="user/gpl", summary="gpl repo",
                license_declared=LicenseKind.GPL_3),
        RepoMap(repo_full_name="user/unk", summary="unknown license",
                license_declared=LicenseKind.UNKNOWN),
    ]


def make_state(tmp_path, plan=None):
    task = Task(task_id="t1", idea="idea", config=TaskConfig())
    state = PipelineState(task=task, repo_maps=repo_maps())
    if plan is not None:
        from ideaforge.state import FusionPlan

        state = state.evolved(fusion_plan=FusionPlan.model_validate(plan))
    return state


# -- plan validity checker ---------------------------------------------------

def test_base_not_in_adopted_rejected():
    plan = dict(GOOD_PLAN, base_repo="user/ghost")
    errors = validate_fusion_plan(_plan(plan), {"user/mit": LicenseKind.MIT})
    assert any("base_repo" in e for e in errors)


def test_donor_not_harvested_rejected():
    plan = json.loads(json.dumps(GOOD_PLAN))
    plan["license_strategy"]["accepted_copyleft"] = True
    errors = validate_fusion_plan(_plan(plan), {"user/mit": LicenseKind.MIT})
    assert any("donor 'user/gpl' is not a harvested" in e for e in errors)


def test_empty_acceptance_rejected():
    plan = json.loads(json.dumps(GOOD_PLAN))
    plan["acceptance"] = []
    from ideaforge.state import FusionPlan

    with pytest.raises(ValidationError):
        FusionPlan.model_validate(plan)


def test_empty_adopt_list_rejected():
    plan = json.loads(json.dumps(GOOD_PLAN))
    plan["donors"][0]["adopt"] = []
    from ideaforge.state import FusionPlan

    with pytest.raises(ValidationError):
        FusionPlan.model_validate(plan)


def test_copyleft_donor_requires_acceptance():
    adopted = {"user/mit": LicenseKind.MIT, "user/gpl": LicenseKind.GPL_3}
    plan = json.loads(json.dumps(GOOD_PLAN))
    plan["license_strategy"]["accepted_copyleft"] = False
    errors = validate_fusion_plan(_plan(plan), adopted)
    assert any("copyleft" in e for e in errors)
    plan["license_strategy"]["accepted_copyleft"] = True
    errors = validate_fusion_plan(_plan(plan), adopted)
    assert errors == []


def _plan(data: dict):
    from ideaforge.state import FusionPlan

    return FusionPlan.model_validate(data)


# -- Architect agent ---------------------------------------------------------

def test_architect_produces_valid_plan(tmp_path):
    pool, _mock = make_pool([json.dumps(GOOD_PLAN)])
    architect = Architect(pool, make_journal(tmp_path, "t1"))
    state = architect.run(make_state(tmp_path))
    assert state.fusion_plan.base_repo == "user/mit"
    assert state.fusion_plan.acceptance[0].item_id == "A1"


def test_architect_retry_exhaustion_fails(tmp_path):
    pool, _mock = make_pool(["garbage"] * 5)
    architect = Architect(pool, make_journal(tmp_path, "t1"))
    with pytest.raises(StructuredOutputError):
        architect.run(make_state(tmp_path))


def test_architect_invalid_plan_raises_agent_failure(tmp_path):
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["base_repo"] = "user/ghost"
    pool, _mock = make_pool([json.dumps(bad)])
    architect = Architect(pool, make_journal(tmp_path, "t1"))
    with pytest.raises(AgentFailure) as exc:
        architect.run(make_state(tmp_path))
    assert "fusion plan invalid" in str(exc.value)


# -- Approval gate ------------------------------------------------------------

def test_approval_approve_transitions_to_running(tmp_path):
    journal = make_journal(tmp_path, "t1")
    state = make_state(tmp_path, GOOD_PLAN)
    task = state.task
    fsm = TaskStateMachine(task, journal)
    gate = ApprovalGate(fsm, journal,
                        lambda payload: ApprovalDecision(decision="approve", gpl_accepted=True))
    out = gate.request(state)
    assert out.fusion_plan.approved is True
    assert task.status == TaskStatus.RUNNING
    events, _ = journal.read_events()
    types = [e.type for e in events]
    assert "approval_required" in types and "approval_received" in types


def test_approval_reject_fails_task(tmp_path):
    journal = make_journal(tmp_path, "t1")
    state = make_state(tmp_path, GOOD_PLAN)
    task = state.task
    fsm = TaskStateMachine(task, journal)
    gate = ApprovalGate(fsm, journal, lambda payload: ApprovalDecision(decision="reject"))
    with pytest.raises(AgentFailure):
        gate.request(state)
    assert task.status == TaskStatus.FAILED


def test_approval_modified_plan_becomes_effective(tmp_path):
    journal = make_journal(tmp_path, "t1")
    state = make_state(tmp_path, GOOD_PLAN)
    task = state.task
    fsm = TaskStateMachine(task, journal)
    modified = _plan(GOOD_PLAN)
    modified.version = 2
    gate = ApprovalGate(
        fsm, journal,
        lambda payload: ApprovalDecision(decision="approve", modified_plan=modified,
                                         gpl_accepted=True),
    )
    out = gate.request(state)
    assert out.fusion_plan.version == 2
    events, _ = journal.read_events()
    received = [e for e in events if e.type == "approval_received"]
    assert any(e.payload.get("modified_plan_used") for e in received)
    assert any(e.payload.get("original_version") == 1 for e in received)


def test_approval_gpl_requires_explicit_acceptance(tmp_path):
    journal = make_journal(tmp_path, "t1")
    plan = json.loads(json.dumps(GOOD_PLAN))
    state = make_state(tmp_path, plan)
    task = state.task
    fsm = TaskStateMachine(task, journal)
    gate = ApprovalGate(
        fsm, journal,
        lambda payload: ApprovalDecision(decision="approve", gpl_accepted=False),
    )
    with pytest.raises(AgentFailure) as exc:
        gate.request(state)
    assert "copyleft" in str(exc.value)
    assert task.status == TaskStatus.FAILED