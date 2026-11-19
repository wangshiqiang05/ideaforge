
import pytest
from mocks import FakeSandboxManager, make_config, make_git_repo, make_journal, prospect

from ideaforge.agents.harvester import Harvester
from ideaforge.errors import AgentFailure, ExecTimeoutError
from ideaforge.state import LicenseKind, PipelineState, ProspectStatus, Task, TaskConfig
from ideaforge.workspace import WorkspaceManager


def make_state(tmp_path, prospects) -> PipelineState:
    task = Task(task_id="t1", idea="fusion idea", config=TaskConfig())
    return PipelineState(task=task, prospects=prospects)


def make_harness(tmp_path, exec_script=None, images=None):
    journal = make_journal(tmp_path, "t1")
    workspace = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    sandbox = FakeSandboxManager(exec_script=exec_script, images=images, journal=journal)
    cfg = make_config()
    harvester = Harvester(workspace, sandbox, journal, cfg)
    return harvester, workspace, sandbox, journal, cfg


def test_harvest_topk_container_install_and_image_commit(tmp_path):
    donor = make_git_repo(tmp_path / "donor_src", {
        "README.md": "# donor", "LICENSE": "MIT License",
        "requirements.txt": "requests\n", "pkg/__init__.py": "",
    })
    state = make_state(tmp_path, [prospect("user/donor", str(donor))])
    harvester, workspace, sandbox, _journal, _cfg = make_harness(tmp_path)
    out = harvester.run(state)
    assert out.prospects[0].status == ProspectStatus.ADOPTED
    assert out.repo_maps[0].repo_full_name == "user/donor"
    assert out.repo_maps[0].license_declared == LicenseKind.MIT
    assert sandbox.committed == ["user_donor"]
    clone_calls = [c for c in sandbox.exec_calls if c[:2] == ["git", "clone"]]
    assert clone_calls, "container-internal clone must happen"
    assert (workspace.forge_dir("t1") / "README.md").exists()


def test_clone_failure_substitutes_next_candidate(tmp_path):
    good = make_git_repo(tmp_path / "good_src", {"README.md": "# good", "LICENSE": "MIT License"})
    state = make_state(tmp_path, [
        prospect("user/broken", str(tmp_path / "does-not-exist")),
        prospect("user/good", str(good)),
    ])
    harvester, _workspace, _sandbox, _journal, _cfg = make_harness(tmp_path)
    out = harvester.run(state)
    adopted = [p for p in out.prospects if p.status == ProspectStatus.ADOPTED]
    assert [p.repo_full_name for p in adopted] == ["user/good"]
    blacklisted = [p for p in out.prospects if p.status == ProspectStatus.BLACKLISTED]
    assert [p.repo_full_name for p in blacklisted] == ["user/broken"]
    assert blacklisted[0].fail_reason and "clone failed" in blacklisted[0].fail_reason


def test_install_exhaustion_fails_task_before_planning(tmp_path):
    donor = make_git_repo(tmp_path / "donor_src", {
        "README.md": "# donor", "LICENSE": "MIT License", "requirements.txt": "x\n",
    })
    state = make_state(tmp_path, [prospect("user/donor", str(donor))])
    timeout = ExecTimeoutError("install timed out")
    harvester, _workspace, _sandbox, journal, _cfg = make_harness(
        tmp_path, exec_script=[timeout, timeout, timeout, timeout]
    )
    with pytest.raises(AgentFailure) as exc:
        harvester.run(state)
    assert "environment setup failed" in str(exc.value)
    events, _ = journal.read_events()
    assert not any(e.type == "approval_required" for e in events)


def test_image_cache_hit_skips_setup(tmp_path):
    donor = make_git_repo(tmp_path / "donor_src", {"README.md": "# donor", "LICENSE": "MIT License"})
    state = make_state(tmp_path, [prospect("user/donor", str(donor))])
    harvester, _workspace, sandbox, _journal, _cfg = make_harness(tmp_path, images={"user_donor"})
    out = harvester.run(state)
    assert out.prospects[0].status == ProspectStatus.ADOPTED
    assert sandbox.exec_calls == []
    assert sandbox.committed == []


def test_forge_first_commit_mirrors_base(tmp_path):
    import git

    donor = make_git_repo(tmp_path / "donor_src", {
        "README.md": "# donor", "LICENSE": "MIT License", "app.py": "print('base')\n",
    })
    state = make_state(tmp_path, [prospect("user/donor", str(donor))])
    harvester, workspace, _sandbox, _journal, _cfg = make_harness(tmp_path)
    harvester.run(state)
    forge = workspace.forge_dir("t1")
    repo = git.Repo(str(forge))
    first = list(repo.iter_commits())[-1]
    assert "mirror of user/donor" in first.message
    assert (forge / "app.py").read_text(encoding="utf-8") == "print('base')\n"


def test_repo_map_in_state_and_license_kinds(tmp_path):
    mit = make_git_repo(tmp_path / "mit_src", {"README.md": "# mit", "LICENSE": "MIT License"})
    unknown = make_git_repo(tmp_path / "unk_src", {"README.md": "# unknown"})
    state = make_state(tmp_path, [
        prospect("user/unk", str(unknown), license_name="UNKNOWN"),
        prospect("user/mit", str(mit), license_name="MIT"),
    ])
    harvester, _workspace, _sandbox, _journal, _cfg = make_harness(tmp_path)
    out = harvester.run(state)
    lic = {m.repo_full_name: m.license_declared for m in out.repo_maps}
    assert lic["user/mit"] == LicenseKind.MIT
    assert lic["user/unk"] == LicenseKind.UNKNOWN