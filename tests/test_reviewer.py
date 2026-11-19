import json

import pytest
from mocks import make_config, make_pool

from ideaforge.agents.reviewer import Reviewer
from ideaforge.errors import AgentFailure
from ideaforge.state import (
    Finding,
    FindingCategory,
    FusionPlan,
    PipelineState,
    ReviewerKind,
    Severity,
    Task,
    TaskConfig,
    Verdict,
    VerdictKind,
)

PASS = json.dumps({"verdict": "PASS", "confidence": 0.95, "findings": []})
FAIL = json.dumps({
    "verdict": "FAIL", "confidence": 0.8,
    "findings": [{"file": "src/x.py", "line": 10, "severity": "major",
                  "category": "style", "description": "bad naming",
                  "suggested_fix": "rename", "item_id": None}],
})


def make_state(with_plan=True) -> PipelineState:
    task = Task(task_id="t1", idea="build a backtest tool", config=TaskConfig())
    plan = None
    if with_plan:
        plan = FusionPlan.model_validate({
            "base_repo": "b",
            "donors": [{"repo": "d", "adopt": [{"path": "a.py"}]}],
            "acceptance": [{"item_id": "A1", "command": "pytest", "description": "runs"}],
        })
    return PipelineState(task=task, fusion_plan=plan,
                         workspace_files={"src/x.py": "def main():\n    pass\n"})


def test_reviewer_pass_verdict():
    pool, _mock = make_pool([PASS])
    reviewer = Reviewer(ReviewerKind.R1_RUNNABLE, pool, None, make_config())
    verdict = reviewer.review(make_state(), 1, [])
    assert verdict.verdict == VerdictKind.PASS
    assert verdict.reviewer == ReviewerKind.R1_RUNNABLE
    assert verdict.cycle == 1


def test_invalid_structure_triggers_corrective_retry():
    pool, mock = make_pool(["definitely not json", FAIL])
    reviewer = Reviewer(ReviewerKind.R2_STYLE, pool, None, make_config())
    verdict = reviewer.review(make_state(), 1, [])
    assert verdict.verdict == VerdictKind.FAIL
    assert len(mock.calls) == 2
    corrective = mock.calls[1]["messages"][-1]
    assert "invalid" in corrective.lower()


def test_retry_exhaustion_fails_with_reviewer_named():
    pool, _mock = make_pool(["garbage"] * 5)
    reviewer = Reviewer(ReviewerKind.R3_IDEA, pool, None, make_config())
    with pytest.raises(AgentFailure) as exc:
        reviewer.review(make_state(), 1, [])
    assert "R3_idea" in str(exc.value)


def test_r3_input_contains_idea_and_acceptance():
    pool, mock = make_pool([PASS])
    reviewer = Reviewer(ReviewerKind.R3_IDEA, pool, None, make_config())
    reviewer.review(make_state(), 1, [])
    user_content = mock.calls[0]["messages"][0]
    assert "build a backtest tool" in user_content
    assert "[A1]" in user_content
    assert "pytest" in user_content


def test_history_verdicts_injected():
    pool, mock = make_pool([PASS])
    reviewer = Reviewer(ReviewerKind.R1_RUNNABLE, pool, None, make_config())
    history = [Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R2_STYLE,
                       confidence=0.7, cycle=0,
                       findings=[Finding(file="src/old.py", line=2, severity=Severity.MINOR,
                                         category=FindingCategory.STYLE, description="old issue")])]
    reviewer.review(make_state(), 1, history)
    content = mock.calls[0]["messages"][0]
    assert "R2_style" in content
    assert "Previous verdict history" in content


def test_all_three_reviewer_kinds_have_distinct_roles():
    from ideaforge.agents.reviewer import ROLE_BY_KIND

    roles = {k: v.value for k, v in ROLE_BY_KIND.items()}
    assert len(set(roles.values())) == 3