import pytest

from ideaforge.errors import PathViolationError
from ideaforge.journal import Journal
from ideaforge.sandbox.safety import blocked_reason, is_blocked, strip_secret_env
from ideaforge.workspace import Patch, WorkspaceManager


def test_denylist_each_category_blocked():
    cases = {
        "rm -rf /": "rm -rf /",
        "mkfs /dev/sda1": "mkfs",
        "dd if=/dev/zero of=/dev/sda": "dd if=/dev/zero",
        ":(){ :|:& };:": ":(){ :|:& };:",
        "shutdown -h now": "shutdown -h",
        "reboot": "reboot",
        "format c: /q": "format c:",
        "conda create -n env python": "conda create",
        "nc -l 8080": "nc -l",
        "systemctl start nginx": "systemctl start",
        "service start docker": "service start",
        "nohup python server.py &": "nohup ",
    }
    for command, expected in cases.items():
        assert blocked_reason(command) == expected, command
        assert is_blocked(command)


def test_normal_commands_pass():
    for command in ("pip install requests", "pytest -q", "python -m compileall .", "ruff check ."):
        assert blocked_reason(command) is None


def test_strip_secret_env():
    env = strip_secret_env({"OPENAI_API_KEY": "sk-1", "HOME": "/h", "PATH": "/bin"})
    assert "OPENAI_API_KEY" not in env
    assert env["HOME"] == "/h"


def test_path_traversal_patch_rejected_with_journal(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    manager = WorkspaceManager(root=tmp_path / "ws", journal=journal)
    manager.init_task("t1")
    (manager.forge_dir("t1")).mkdir(parents=True, exist_ok=True)

    with pytest.raises(PathViolationError):
        manager.apply_patch("t1", Patch(files={"../../escape.py": "x"}))
    events, _ = journal.read_events()
    assert any(e.type == "patch_rejected" for e in events)

    with pytest.raises(PathViolationError):
        manager.apply_patch("t1", Patch(files={"/abs/path.py": "x"}))
    result = manager.apply_patch("t1", Patch(files={"ok.py": "print(1)"}))
    assert result.applied == ["ok.py"]
    assert result.rejected == []