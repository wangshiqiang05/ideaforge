import json

import pytest
from mocks import make_config, make_git_repo, make_journal, make_pool

from ideaforge.agents.synthesizer import (
    Synthesizer,
    rewrite_imports,
    syntax_ok,
)
from ideaforge.errors import AgentFailure
from ideaforge.state import (
    ChangeSource,
    FusionPlan,
    PipelineState,
    Task,
    TaskConfig,
)
from ideaforge.workspace import WorkspaceManager


def make_plan(base="user/base", donor="user/donor", symbol=None) -> FusionPlan:
    return FusionPlan.model_validate({
        "base_repo": base,
        "donors": [{"repo": donor, "adopt": [{"path": "mylib/util.py", "symbol": symbol}],
                    "drop": [], "conflicts": []}],
        "acceptance": [{"item_id": "A1", "command": "python -c 'print(1)'"}],
        "rename_map": {"mylib": "forgelib"},
    })


def test_rewrite_imports_exact_and_prefix():
    src = "import mylib.util\nfrom mylib import helper\nimport os\n"
    out = rewrite_imports(src, {"mylib": "forgelib"})
    assert "forgelib.util" in out
    assert "from forgelib import helper" in out
    assert "import os" in out
    assert syntax_ok(out)
    assert rewrite_imports("import other\n", {"mylib": "forgelib"}) == "import other\n"


def test_rewrite_nested_prefix():
    src = "from mylib.util.sub import thing\n"
    out = rewrite_imports(src, {"mylib": "forgelib"})
    assert "forgelib.util.sub" in out


def test_full_synthesis_flow(tmp_path):
    donor = make_git_repo(tmp_path / "donor", {
        "mylib/util.py": "import mylib.helper\n\ndef run():\n    return 1\n",
        "mylib/helper.py": "H = 1\n",
    })
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    workspace.harvest_repo("t1", "user/donor", str(donor))
    workspace.init_forge("t1", "user/donor")

    cfg = make_config()
    pool, _mock = make_pool([json.dumps({"files": {}})], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    state = PipelineState(task=Task(task_id="t1", idea="idea", config=TaskConfig()))
    plan = make_plan(symbol="run")
    plan.approved = True
    state = state.evolved(fusion_plan=plan)

    out = synth.run(state)
    forge = workspace.forge_dir("t1")
    migrated = forge / "forgelib" / "util.py"
    assert migrated.exists()
    content = migrated.read_text(encoding="utf-8")
    assert "forgelib.helper" in content
    assert "mylib" not in content
    sources = {c.source for c in out.changelog}
    assert ChangeSource.DONOR_REPO in sources
    # spec 5.5.3.a: change records must carry target file, function identifier
    # and source repo for function-level transplants
    donor_record = next(c for c in out.changelog if c.source == ChangeSource.DONOR_REPO)
    assert "user/donor" in donor_record.reason
    assert "mylib/util.py" in donor_record.reason
    assert "run" in donor_record.reason
    assert out.workspace_files, "forge scan must populate workspace_files"


def test_unapproved_plan_rejected(tmp_path):
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    cfg = make_config()
    pool, _ = make_pool([], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    state = PipelineState(task=Task(task_id="t1", idea="i", config=TaskConfig()),
                          fusion_plan=make_plan())
    with pytest.raises(AgentFailure):
        synth.run(state)


def test_critical_missing_donor_module_fails(tmp_path):
    donor = make_git_repo(tmp_path / "donor", {"README.md": "# d"})
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    workspace.harvest_repo("t1", "user/donor", str(donor))
    workspace.init_forge("t1", "user/donor")
    cfg = make_config()
    pool, _ = make_pool([], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    plan = FusionPlan.model_validate({
        "base_repo": "user/donor",
        "donors": [{"repo": "user/donor", "adopt": [{"path": "mylib/util.py", "symbol": "run"}]}],
        "acceptance": [{"item_id": "A1", "command": "true"}],
    })
    plan.approved = True
    state = PipelineState(task=Task(task_id="t1", idea="i", config=TaskConfig()),
                          fusion_plan=plan)
    with pytest.raises(AgentFailure) as exc:
        synth.run(state)
    assert "critical donor module" in str(exc.value)


def test_non_critical_missing_donor_module_skipped(tmp_path):
    donor = make_git_repo(tmp_path / "donor", {"README.md": "# d"})
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    workspace.harvest_repo("t1", "user/donor", str(donor))
    workspace.init_forge("t1", "user/donor")
    cfg = make_config()
    pool, _ = make_pool([], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    plan = FusionPlan.model_validate({
        "base_repo": "user/donor",
        "donors": [{"repo": "user/donor", "adopt": [{"path": "missing.py", "symbol": None}]}],
        "acceptance": [{"item_id": "A1", "command": "true"}],
    })
    plan.approved = True
    state = PipelineState(task=Task(task_id="t1", idea="i", config=TaskConfig()),
                          fusion_plan=plan)
    synth.run(state)
    events, _ = journal.read_events()
    assert any(e.type == "patch_rejected" for e in events)


def test_repos_untouched_after_synthesis(tmp_path):
    donor = make_git_repo(tmp_path / "donor", {
        "mylib/util.py": "def run():\n    return 1\n",
    })
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    workspace.harvest_repo("t1", "user/donor", str(donor))
    workspace.init_forge("t1", "user/donor")
    cfg = make_config()
    pool, _ = make_pool([], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    plan = make_plan()
    plan.approved = True
    state = PipelineState(task=Task(task_id="t1", idea="i", config=TaskConfig()),
                          fusion_plan=plan)
    synth.run(state)
    workspace.assert_repos_untouched("t1")  # must not raise


def test_bridge_generation_recorded(tmp_path):
    donor = make_git_repo(tmp_path / "donor", {"mylib/util.py": "def run():\n    return 1\n"})
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    workspace.init_task("t1")
    workspace.harvest_repo("t1", "user/donor", str(donor))
    workspace.init_forge("t1", "user/donor")
    cfg = make_config()
    bridge_files = {"bridge.py": "from forgelib.util import run\n\nBRIDGE = True\n"}
    pool, _ = make_pool([json.dumps({"files": bridge_files})], cfg=cfg)
    synth = Synthesizer(pool, workspace, journal)
    plan = FusionPlan.model_validate({
        "base_repo": "user/donor",
        "donors": [{"repo": "user/donor", "adopt": [{"path": "mylib/util.py", "symbol": None}]}],
        "bridge_design": [{"name": "bridge.py", "description": "wiring"}],
        "acceptance": [{"item_id": "A1", "command": "true"}],
        "rename_map": {"mylib": "forgelib"},
    })
    plan.approved = True
    state = PipelineState(task=Task(task_id="t1", idea="i", config=TaskConfig()),
                          fusion_plan=plan)
    out = synth.run(state)
    assert (workspace.forge_dir("t1") / "bridge.py").exists()
    sources = {c.source for c in out.changelog}
    assert ChangeSource.BRIDGE_GEN in sources