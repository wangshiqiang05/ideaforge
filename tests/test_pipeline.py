"""End-to-end orchestrator test: real pipeline with fake GitHub/LLM/Docker.

Covers tasks.md 18.1: six stages in order, failure propagation to failed,
complete event sequence, and full delivery artefacts."""
import json

import pytest
from mocks import (
    FakeHTTP,
    FakeSandboxManager,
    make_config,
    make_git_repo,
    make_journal,
    make_pool,
)

from ideaforge.agents.architect import ApprovalDecision, ApprovalGate
from ideaforge.errors import AgentFailure
from ideaforge.pipeline import PipelineOrchestrator
from ideaforge.state import (
    PipelineState,
    Task,
    TaskConfig,
    TaskStateMachine,
    TaskStatus,
)
from ideaforge.workspace import WorkspaceManager

QUERIES = json.dumps({"queries": [{"text": "demo tool"}, {"topic": "demo"}]})
SCORE = json.dumps({"relevance": 5, "modifiability": 5, "quality": 4,
                    "license_compat": 5, "license": "MIT"})
PASS = json.dumps({"verdict": "PASS", "confidence": 0.95, "findings": []})


def build_pipeline(tmp_path, scripts, decision_hook=None):
    donor = make_git_repo(tmp_path / "donor_src", {
        "README.md": "# demo donor tool\nrun with python",
        "LICENSE": "MIT License",
        "mylib/util.py": "def run():\n    return 42\n",
        "mylib/__init__.py": "",
    })
    journal = make_journal(tmp_path, "t-e2e")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    sandbox = FakeSandboxManager(journal=journal)
    cfg = make_config(max_cycles=3)
    pool, mock = make_pool(scripts, cfg=cfg)

    http = FakeHTTP({"demo tool": [
        {"full_name": "user/donor", "clone_url": str(donor), "description": "demo"},
    ], "topic:demo": []})
    from ideaforge.agents.prospector import Prospector

    prospector = Prospector(pool, journal, cfg, github_token="ghp_e2etoken1234567890", http=http)
    from ideaforge.agents.harvester import Harvester

    harvester = Harvester(workspace, sandbox, journal, cfg)
    from ideaforge.agents.architect import Architect

    architect = Architect(pool, journal)
    task = Task(task_id="t-e2e", idea="fuse a demo tool",
                config=TaskConfig(max_cycles=3, budget_usd=100.0))
    state = PipelineState(task=task)
    fsm = TaskStateMachine(task, journal)
    approval = ApprovalGate(fsm, journal, decision_hook or (lambda payload: ApprovalDecision(decision="approve")))
    from ideaforge.agents.synthesizer import Synthesizer

    synthesizer = Synthesizer(pool, workspace, journal)
    from ideaforge.forge.gates import ToolGate

    gate = ToolGate(sandbox, journal, cfg)
    from ideaforge.agents.fixer import Fixer
    from ideaforge.agents.judge import JudgeFinal
    from ideaforge.agents.reviewer import Reviewer, ReviewerKind
    from ideaforge.delivery import Delivery
    from ideaforge.forge.loop import ForgeLoopController

    reviewers = [Reviewer(k, pool, journal, cfg) for k in
                 (ReviewerKind.R1_RUNNABLE, ReviewerKind.R2_STYLE, ReviewerKind.R3_IDEA)]
    forge_loop = ForgeLoopController(gate, reviewers, JudgeFinal(pool, journal, cfg),
                                     Fixer(pool, journal, cfg), workspace, sandbox,
                                     pool.ledger, journal, cfg, fsm, decision_hook)
    delivery = Delivery(sandbox, journal, cfg)
    orch = PipelineOrchestrator(prospector, harvester, architect, approval, synthesizer,
                                forge_loop, delivery, workspace, sandbox, journal,
                                fsm, pool.ledger, cfg)
    return orch, state, fsm, journal, workspace, mock


def test_six_stage_pipeline_completes(tmp_path):
    plan = {
        "base_repo": "user/donor",
        "donors": [{"repo": "user/donor",
                    "adopt": [{"path": "mylib/util.py", "symbol": None}],
                    "drop": [], "conflicts": []}],
        "bridge_design": [],
        "acceptance": [{"item_id": "A1", "command": "python -c 'print(42)'",
                        "description": "smoke"}],
        "rename_map": {"mylib": "forgelib"},
        "license_strategy": {"accepted_copyleft": False, "notice": ""},
    }
    scripts = [QUERIES, SCORE, json.dumps(plan), PASS, PASS, PASS, PASS]
    orch, state, _fsm, journal, workspace, _mock = build_pipeline(tmp_path, scripts)
    out = orch.run(state)

    assert out.task.status == TaskStatus.COMPLETED
    assert out.fusion_plan is not None and out.fusion_plan.approved
    assert (workspace.forge_dir("t-e2e") / "forgelib" / "util.py").exists()

    events, _ = journal.read_events()
    enter_stages = [e.payload["stage"] for e in events if e.type == "stage_enter"]
    assert enter_stages == ["Prospecting", "Harvesting", "Planning", "Synthesis",
                            "ForgeLoop", "Delivery"]
    exit_events = [e for e in events if e.type == "stage_exit"]
    assert all(not e.payload.get("failed") for e in exit_events)
    assert "completed" in [e.type for e in events]

    report = json.loads((tmp_path / "ws" / "t-e2e" / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] is True
    assert report["acceptance_done"] == 1
    assert report["cost"]["total_usd"] > 0
    notice = (tmp_path / "ws" / "t-e2e" / "NOTICE.md").read_text(encoding="utf-8")
    assert "user/donor" in notice
    blob = (tmp_path / "t-e2e" / "journal.jsonl").read_text(encoding="utf-8")
    assert "ghp_e2etoken1234567890" not in blob


def test_stage_failure_propagates_to_failed(tmp_path):
    scripts = [QUERIES, "garbage", "garbage", "garbage", "garbage", "garbage"]
    orch, state, fsm, journal, _workspace, _mock = build_pipeline(tmp_path, scripts)
    with pytest.raises(AgentFailure):
        orch.run(state)
    assert fsm.task.status == TaskStatus.FAILED
    events, _ = journal.read_events()
    failed = [e for e in events if e.type == "failed"]
    assert failed and failed[0].payload["reason"]
    assert any(e.type == "stage_exit" and e.payload.get("failed")
               for e in events)
    report = json.loads((tmp_path / "ws" / "t-e2e" / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] is False