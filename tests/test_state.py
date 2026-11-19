import pytest
from pydantic import ValidationError

from ideaforge.state import (
    Finding,
    FindingCategory,
    FusionPlan,
    GateStage,
    LicenseKind,
    PipelineState,
    ReviewerKind,
    ScoreCard,
    Severity,
    Task,
    TaskStatus,
    Verdict,
    VerdictKind,
)


def test_task_rejects_empty_idea():
    with pytest.raises(ValidationError):
        Task(task_id="t", idea="   ")


def test_task_rejects_overlong_idea():
    with pytest.raises(ValidationError):
        Task(task_id="t", idea="x" * 5001)


def test_task_accepts_boundary_idea():
    task = Task(task_id="t", idea="x" * 5000)
    assert task.status == TaskStatus.RUNNING


def test_invalid_enum_value_rejected():
    with pytest.raises(ValueError):
        TaskStatus("bogus")
    with pytest.raises(ValueError):
        VerdictKind("MAYBE")
    with pytest.raises(ValueError):
        GateStage("fluff")


def test_score_card_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ScoreCard(relevance=6, modifiability=1, quality=1, license_compat=1)
    with pytest.raises(ValidationError):
        ScoreCard(relevance=-1, modifiability=1, quality=1, license_compat=1)


def test_score_card_total_with_weights():
    card = ScoreCard(relevance=4, modifiability=2, quality=0, license_compat=5,
                     license=LicenseKind.MIT)
    assert card.total() == pytest.approx((4 + 2 + 0 + 5) / 4)
    assert card.total({"relevance": 1.0, "modifiability": 0, "quality": 0,
                       "license_compat": 0}) == pytest.approx(4.0)


def test_fail_verdict_requires_findings():
    with pytest.raises(ValidationError):
        Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R1_RUNNABLE,
                confidence=0.5, findings=[])
    v = Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R1_RUNNABLE,
                confidence=0.5, findings=[
                    Finding(file="a.py", line=1, severity=Severity.MAJOR,
                            category=FindingCategory.STYLE, description="x")
                ])
    assert v.verdict == VerdictKind.FAIL


def test_confidence_range_enforced():
    with pytest.raises(ValidationError):
        Verdict(verdict=VerdictKind.PASS, reviewer=ReviewerKind.R1_RUNNABLE,
                confidence=1.5, findings=[])


def test_finding_file_must_be_relative():
    with pytest.raises(ValidationError):
        Finding(file="/abs/path.py", severity=Severity.MINOR,
                category=FindingCategory.OTHER, description="x")
    with pytest.raises(ValidationError):
        Finding(file="../escape.py", severity=Severity.MINOR,
                category=FindingCategory.OTHER, description="x")
    ok = Finding(file="src/ok.py", severity=Severity.MINOR,
                 category=FindingCategory.OTHER, description="x")
    assert ok.file == "src/ok.py"


def test_idea_match_finding_needs_item_id():
    with pytest.raises(ValidationError):
        Finding(file="a.py", severity=Severity.MAJOR,
                category=FindingCategory.IDEA_MATCH, description="missing acceptance")
    ok = Finding(file="a.py", severity=Severity.MAJOR,
                 category=FindingCategory.IDEA_MATCH, description="missing", item_id="A1")
    assert ok.item_id == "A1"


def test_fusion_plan_rename_map_non_empty():
    with pytest.raises(ValidationError):
        FusionPlan(
            base_repo="b",
            donors=[{"repo": "d", "adopt": [{"path": "a.py"}]}],
            acceptance=[{"item_id": "A1", "command": "pytest"}],
            rename_map={"": "x"},
        )


def test_state_evolution_returns_new_instance():
    task = Task(task_id="t", idea="idea")
    state = PipelineState(task=task)
    evolved = state.evolved(workspace_files={"a.py": "print(1)"})
    assert evolved is not state
    assert state.workspace_files == {}
    assert evolved.workspace_files == {"a.py": "print(1)"}
    snap = evolved.snapshot()
    snap.workspace_files["a.py"] = "mutated"
    assert evolved.workspace_files["a.py"] == "print(1)"