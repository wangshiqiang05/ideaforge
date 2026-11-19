import json

import pytest
from mocks import make_config, make_journal, make_pool

from ideaforge.agents.fixer import Fixer, FixInput
from ideaforge.errors import AgentFailure
from ideaforge.state import (
    Finding,
    FindingCategory,
    FusionPlan,
    PipelineState,
    Severity,
    Task,
    TaskConfig,
    ToolReport,
)

FIX = json.dumps({"files": {"src/x.py": "def main():\n    return 1\n"}})


def make_state() -> PipelineState:
    task = Task(task_id="t1", idea="the original idea", config=TaskConfig())
    plan = FusionPlan.model_validate({
        "base_repo": "b",
        "donors": [{"repo": "d", "adopt": [{"path": "a.py"}]}],
        "acceptance": [{"item_id": "A1", "command": "pytest"}],
    })
    return PipelineState(task=task, fusion_plan=plan)


def make_findings() -> list[Finding]:
    return [Finding(file="src/x.py", line=1, severity=Severity.MAJOR,
                    category=FindingCategory.STYLE, description="bad style")]


def make_tool_report() -> ToolReport:
    return ToolReport(passed=False, failed_stage="test",
                      raw_output="FAILED tests/test_x.py - assert 1 == 2")


def test_input_assembler_contains_four_fields():
    pool, _mock = make_pool([FIX])
    fixer = Fixer(pool, make_journal(__import__("pathlib").Path(__file__).parent / "ws-fix", "t1"),
                  make_config())
    fix_input = fixer.build_input(make_state(), make_findings(), make_tool_report())
    missing = fix_input.missing_fields()
    assert missing == []
    assert "the original idea" in fix_input.task_and_plan
    assert "bad style" in fix_input.findings_verbatim
    assert "assert 1 == 2" in fix_input.tool_output_verbatim


def test_empty_findings_refuse_fix_round():
    pool, mock = make_pool([FIX])
    fixer = Fixer(pool, None, make_config())
    with pytest.raises(AgentFailure):
        fixer.build_input(make_state(), [], make_tool_report())
    assert mock.calls == []


def test_missing_field_refuses_call():
    fix_input = FixInput(task_and_plan="", previous_changes="x",
                         findings_verbatim="y", tool_output_verbatim="z")
    assert "task_and_plan" in fix_input.missing_fields()
    pool, mock = make_pool([FIX])
    fixer = Fixer(pool, None, make_config())
    with pytest.raises(AgentFailure):
        fixer.fix(make_state(), 1, fix_input)
    assert mock.calls == []


def test_fixer_consumes_all_feedback_in_prompt(tmp_path):
    pool, mock = make_pool([FIX])
    journal = make_journal(tmp_path, "t1")
    fixer = Fixer(pool, journal, make_config())
    fix_input = fixer.build_input(make_state(), make_findings(), make_tool_report())
    result = fixer.fix(make_state(), 1, fix_input)
    assert result.valid is True
    prompt = mock.calls[0]["messages"][0]
    assert "the original idea" in prompt
    assert "bad style" in prompt
    assert "assert 1 == 2" in prompt
    # journal records the four input digests (replay-verifiable consumption)
    events, _ = journal.read_events()
    fix_events = [e for e in events if e.type == "fix_applied"]
    assert fix_events and len(fix_events[0].payload.get("input_digests", {})) == 4


def test_syntactically_broken_output_rejected():
    pool, _ = make_pool([json.dumps({"files": {"broken.py": "def f(:\n  pass"}})])
    fixer = Fixer(pool, None, make_config())
    fix_input = fixer.build_input(make_state(), make_findings(), make_tool_report())
    result = fixer.fix(make_state(), 1, fix_input)
    assert result.valid is False


def test_digests_stable_for_replay():
    fix_input = FixInput(task_and_plan="a", previous_changes="b",
                         findings_verbatim="c", tool_output_verbatim="d")
    d1 = fix_input.digests()
    d2 = fix_input.digests()
    assert d1 == d2
    assert all(len(v) == 16 for v in d1.values())