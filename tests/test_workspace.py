import pytest

from ideaforge.errors import PathViolationError, RepoTamperedError
from ideaforge.journal import Journal
from ideaforge.workspace import Patch, WorkspaceManager


def make_git_repo(path, files: dict[str, str]) -> None:
    import git

    repo = git.Repo.init(str(path), initial_branch="main")
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Tester")
        cw.set_value("user", "email", "t@t.local")
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    repo.git.add(A=True)
    repo.index.commit("init")


@pytest.fixture()
def base_repo(tmp_path):
    repo = tmp_path / "src_repo"
    repo.mkdir()
    make_git_repo(repo, {"app.py": "print('base')\n", "lib/core.py": "X = 1\n"})
    return repo


def test_task_isolation(tmp_path, base_repo):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    manager.init_task("t2")
    manager.forge_dir("t1").mkdir(parents=True, exist_ok=True)
    manager.forge_dir("t2").mkdir(parents=True, exist_ok=True)
    (manager.forge_dir("t1") / "only_t1.txt").write_text("1", encoding="utf-8")
    (manager.forge_dir("t2") / "only_t2.txt").write_text("2", encoding="utf-8")
    assert (manager.forge_dir("t1") / "only_t1.txt").exists()
    assert not (manager.forge_dir("t2") / "only_t1.txt").exists()


def test_harvest_and_init_forge_first_commit_matches_base(tmp_path, base_repo):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    manager.harvest_repo("t1", "owner/src", str(base_repo))
    forge = manager.init_forge("t1", "owner/src")
    assert (forge / "app.py").read_text(encoding="utf-8") == "print('base')\n"
    assert not (forge / ".git").exists() or (forge / ".git").exists()

    import git

    repo = git.Repo(str(forge))
    first_commit = list(repo.iter_commits())[-1]
    assert "mirror of owner/src" in first_commit.message


def test_commit_and_diff_against_base(tmp_path, base_repo):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    manager.harvest_repo("t1", "owner/src", str(base_repo))
    manager.init_forge("t1", "owner/src")
    result = manager.apply_patch("t1", Patch(files={"new.py": "Y = 2\n"}))
    assert result.applied == ["new.py"]
    sha = manager.commit_forge("t1", "add new module")
    assert sha
    diff = manager.diff_against_base("t1")
    assert "new.py" in diff


def test_path_violation_patch_discarded(tmp_path, base_repo):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")

    (manager.forge_dir("t1")).mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathViolationError):
        manager.apply_patch("t1", Patch(files={"../outside.py": "x"}))
    events, _ = journal.read_events()
    assert any(e.type == "patch_rejected" for e in events)


def test_repo_tampering_detected(tmp_path, base_repo):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    manager.harvest_repo("t1", "owner/src", str(base_repo))
    manager.assert_repos_untouched("t1")

    victim = manager.repos_dir("t1") / "owner_src" / "app.py"
    victim.write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(RepoTamperedError):
        manager.assert_repos_untouched("t1")


def test_partial_rejection_keeps_valid_files(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    (manager.forge_dir("t1")).mkdir(parents=True, exist_ok=True)
    result = manager.apply_patch(
        "t1", Patch(files={"good.py": "A=1\n", "../bad.py": "B=2\n"})
    )
    assert result.applied == ["good.py"]
    assert result.rejected == ["../bad.py"]