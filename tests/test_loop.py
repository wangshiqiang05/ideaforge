"""Forge Loop regression: convergence / degradation / oscillation paths,
critical & budget suspension-resume, feedback consumption and fix rollback
(design 2.8.3.1-2, tasks 16-17)."""
import json

import pytest
from mocks import (
    ExecResult,
    FakeSandboxManager,
    NoCache,
    finding,
    make_config,
    make_journal,
    make_pool,
    verdict_json,
)

from ideaforge.agents.fixer import Fixer
from ideaforge.agents.judge import JudgeFinal
from ideaforge.agents.reviewer import Reviewer, ReviewerKind
from ideaforge.forge.gates import ToolGate
from ideaforge.forge.loop import BudgetDecision, ForgeLoopController
from ideaforge.state import (
    PipelineState,
    Task,
    TaskConfig,
    TaskStateMachine,
    TaskStatus,
)
from ideaforge.workspace import WorkspaceManager

PASS = verdict_json(True)
FAIL_STABLE = verdict_json(False, [
    finding(file="src/x.py", line=1, severity="major", category="style", description="bug A")
])
FAIL_CRITICAL = verdict_json(False, [
    finding(file="src/x.py", line=3, severity="critical", category="runnable",
            description="security hole in auth flow")
])
FIX_EMPTY = json.dumps({"files": {}})
FIX_OK = json.dumps({"files": {"src/x.py": "def fixed():\n    return 1\n"}})
FIX_BROKEN = json.dumps({"files": {"broken.py": "def f(:"}})


def make_loop(tmp_path, scripts, exec_script=None, decision_hook=None,
              max_cycles=6, budget=100.0):
    journal = make_journal(tmp_path, "t-loop")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t-loop")
    workspace.forge_dir("t-loop").mkdir(exist_ok=True)
    sandbox = FakeSandboxManager(exec_script=exec_script, journal=journal)
    cfg = make_config(max_cycles=max_cycles, budget=budget)
    pool, mock = make_pool(scripts, cfg=cfg)
    pool.cache = NoCache()  # deterministic script sequencing across rounds
    gate = ToolGate(sandbox, journal, cfg)
    reviewers = [
        Reviewer(kind, pool, journal, cfg)
        for kind in (ReviewerKind.R1_RUNNABLE, ReviewerKind.R2_STYLE, ReviewerKind.R3_IDEA)
    ]
    judge = JudgeFinal(pool, journal, cfg)
    fixer = Fixer(pool, journal, cfg)
    task = Task(task_id="t-loop", idea="forge an idea",
                config=TaskConfig(max_cycles=max_cycles, budget_usd=budget))
    state = PipelineState(task=task)
    fsm = TaskStateMachine(task, journal)
    loop = ForgeLoopController(gate, reviewers, judge, fixer, workspace, sandbox,
                               pool.ledger, journal, cfg, fsm, decision_hook)
    return loop, state, fsm, journal, mock, workspace, sandbox


def test_convergence_path_completes(tmp_path):
    loop, state, _fsm, journal, _mock, _, _ = make_loop(tmp_path, [PASS, PASS, PASS, PASS])
    out = loop.run(state)
    assert out.task.status == TaskStatus.COMPLETED
    assert len(out.cycles) == 1
    assert out.cycles[0].tool_report.passed is True
    assert len(out.cycles[0].verdicts) == 4  # three reviewers + final judge
    assert out.cost.total_usd > 0
    events, _ = journal.read_events()
    types = [e.type for e in events]
    assert "completed" in types
    assert types.count("verdict_produced") == 4
    assert types.count("cycle_start") == 1
    assert types.count("tool_gate_result") == 1
    assert "fix_applied" not in types  # converged without needing fixes


def test_degradation_path_after_max_cycles(tmp_path):
    gate_fail = ExecResult(success=False, exit_code=1, output="SyntaxError: invalid syntax")
    loop, state, _fsm, journal, mock, _, _ = make_loop(
        tmp_path, [FIX_EMPTY, FIX_EMPTY], exec_script=[gate_fail, gate_fail], max_cycles=2
    )
    out = loop.run(state)
    assert out.task.status == TaskStatus.DEGRADED
    assert out.task.cycle == 2
    assert all(c.tool_report.passed is False for c in out.cycles)
    assert all(c.verdicts == [] for c in out.cycles)  # tool-gate short circuit: zero LLM reviews
    events, _ = journal.read_events()
    assert "degraded" in [e.type for e in events]
    # fixer still consumed the tool output verbatim (two rounds)
    fix_calls = [c for c in mock.calls if "Tool-gate output" in (c["messages"][0] if c["messages"] else "")]
    assert fix_calls


def test_oscillation_escalates_then_degrades(tmp_path):
    # Per-round call sequence: R1, R2, R3, fixer -> repeat for 4 cycles.
    scripts = ([FAIL_STABLE] * 3 + [FIX_EMPTY]) * 4
    loop, state, _fsm, journal, _mock, _, _ = make_loop(tmp_path, scripts, max_cycles=4)
    out = loop.run(state)
    assert out.task.status == TaskStatus.DEGRADED
    oscillations = [c.oscillation for c in out.cycles]
    assert oscillations == [False, False, True, True]
    escalated_round = out.cycles[2]
    assert escalated_round.oscillation is True
    events, _ = journal.read_events()
    assert "degraded" in [e.type for e in events]


def test_critical_pause_approved_resumes(tmp_path):
    decisions = []

    def hook(payload):
        decisions.append(payload)
        from ideaforge.agents.architect import ApprovalDecision

        return ApprovalDecision(decision="approve")

    scripts = [PASS, PASS, FAIL_CRITICAL, FIX_OK, PASS, PASS, PASS, PASS]
    loop, state, _fsm, journal, _mock, _, _ = make_loop(tmp_path, scripts, max_cycles=2,
                                                      decision_hook=hook)
    out = loop.run(state)
    assert out.task.status == TaskStatus.COMPLETED
    assert decisions and decisions[0]["kind"] == "critical"
    assert out.task.cycle == 2  # resumed and finished in round 2
    events, _ = journal.read_events()
    types = [e.type for e in events]
    assert "finding_critical" in types
    assert "approval_received" in types


def test_critical_pause_rejected_fails_task(tmp_path):
    from ideaforge.agents.architect import ApprovalDecision

    def hook(payload):
        return ApprovalDecision(decision="reject")

    loop, state, fsm, _journal, _, _, _ = make_loop(
        tmp_path, [PASS, PASS, FAIL_CRITICAL], max_cycles=2, decision_hook=hook
    )
    with pytest.raises(Exception) as exc:
        loop.run(state)
    assert "aborted" in str(exc.value)
    assert fsm.task.status == TaskStatus.FAILED


def test_budget_suspension_before_next_model_call(tmp_path):
    decisions = []

    def hook(payload):
        decisions.append(payload)
        return BudgetDecision(decision="raise_budget", new_budget=10.0)

    loop, state, _fsm, journal, _mock, _, _ = make_loop(
        tmp_path, [PASS, PASS, PASS, PASS], max_cycles=1, budget=0.0, decision_hook=hook
    )
    out = loop.run(state)
    assert out.task.status == TaskStatus.COMPLETED
    assert decisions and decisions[0]["kind"] == "budget"
    assert loop.ledger.budget_usd == 10.0
    events, _ = journal.read_events()
    assert "budget_break" in [e.type for e in events]


def test_budget_abort_fails_task(tmp_path):
    def hook(payload):
        return BudgetDecision(decision="abort")

    loop, state, fsm, _journal, _, _, _ = make_loop(
        tmp_path, [PASS], max_cycles=1, budget=0.0, decision_hook=hook
    )
    with pytest.raises(Exception) as exc:
        loop.run(state)
    assert "aborted" in str(exc.value)
    assert fsm.task.status == TaskStatus.FAILED


def test_budget_invalid_raise_resuspends(tmp_path):
    """review_fix spec 5.3.1.3: a raise that is not above the current budget
    must not silently resume the loop; the task re-suspends until a legal
    decision or an abort arrives, so no unbreakable pause loop exists."""
    responses = [
        BudgetDecision(decision="raise_budget"),                 # new_budget None: invalid
        BudgetDecision(decision="raise_budget", new_budget=0.0), # not above 0.0: invalid
        BudgetDecision(decision="raise_budget", new_budget=10.0),# legal raise
    ]
    calls = {"n": 0}
    decisions = []

    def hook(payload):
        decisions.append(payload)
        resp = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return resp

    loop, state, _fsm, journal, _mock, _, _ = make_loop(
        tmp_path, [PASS, PASS, PASS, PASS], max_cycles=1, budget=0.0, decision_hook=hook
    )
    out = loop.run(state)
    assert out.task.status == TaskStatus.COMPLETED
    assert loop.ledger.budget_usd == 10.0
    assert calls["n"] == 3  # two invalid raises were re-asked, not silently consumed
    breaks = [e for e in journal.read_events()[0] if e.type == "budget_break"]
    assert len(breaks) == 3  # initial suspension + one re-suspension per invalid raise


def test_feedback_consumption_verbatim_in_prompt(tmp_path):
    """spec 5.6.1.7 made executable: the fixer prompt must contain the
    aggregated findings and tool-gate output verbatim."""
    gate_fail = ExecResult(success=False, exit_code=1, output="SyntaxError: name 'x' is not defined")
    loop, state, _fsm, journal, mock, _, _ = make_loop(
        tmp_path, [FIX_EMPTY, FIX_EMPTY], exec_script=[gate_fail, gate_fail], max_cycles=2
    )
    loop.run(state)
    fixer_prompts = [c["messages"][0] for c in mock.calls if "Tool-gate output" in c["messages"][0]]
    assert fixer_prompts, "fixer must be called with assembled feedback"
    for prompt in fixer_prompts:
        assert "SyntaxError" in prompt          # tool output verbatim
        assert "toolgate" in prompt             # aggregated findings source
        assert "Your previous changes" in prompt
        assert "forge an idea" in prompt        # task + plan section
    events, _ = journal.read_events()
    digests = [e.payload["input_digests"] for e in events if e.type == "fix_applied"]
    assert digests and all(len(d) == 4 for d in digests)


def test_invalid_fix_output_rolls_back(tmp_path):
    gate_fail = ExecResult(success=False, exit_code=1, output="SyntaxError: broken")
    loop, state, _fsm, journal, _mock, workspace, _ = make_loop(
        tmp_path, [FIX_BROKEN, FIX_BROKEN], exec_script=[gate_fail, gate_fail], max_cycles=2
    )
    out = loop.run(state)
    assert out.task.status == TaskStatus.DEGRADED
    assert out.task.cycle == 2  # rounds still counted
    for record in out.cycles:
        assert record.fix_diff is None  # nothing applied
    assert not (workspace.forge_dir("t-loop") / "broken.py").exists()
    events, _ = journal.read_events()
    assert any(e.type == "fix_rejected" for e in events)


def test_cycle_bounds_enforced(tmp_path):
    loop, state, _fsm, _journal, _mock, _, _ = make_loop(
        tmp_path, [PASS, PASS, PASS, PASS], max_cycles=6
    )
    out = loop.run(state)
    assert 1 <= out.task.cycle <= 6