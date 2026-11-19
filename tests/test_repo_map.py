from pathlib import Path

from ideaforge.agents.repo_map import (
    build_repo_map,
    detect_license,
    estimate_tokens,
)
from ideaforge.llm.reasoning import strip_reasoning_content
from ideaforge.state import LicenseKind


def test_estimate_tokens_bound():
    text = "word " * 1000  # ~5000 chars
    assert estimate_tokens(text) <= 5000 // 4 + 1


def make_repo(tmp_path: Path, license_text: str | None, files: int = 30) -> Path:
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(files):
        (pkg / f"mod{i}.py").write_text(f"# module {i}\n", encoding="utf-8")
    (repo / "README.md").write_text("# demo", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    if license_text:
        (repo / "LICENSE").write_text(license_text, encoding="utf-8")
    return repo


def test_repo_map_within_token_budget(tmp_path):
    repo = make_repo(tmp_path, None, files=60)
    rm = build_repo_map("owner/demo", repo, token_budget=500)
    assert estimate_tokens(rm.summary) <= 500
    assert rm.total_files > 50
    assert rm.license_declared == LicenseKind.UNKNOWN
    assert "pkg" in rm.summary


def test_repo_map_key_files_and_languages(tmp_path):
    repo = make_repo(tmp_path, None, files=5)
    rm = build_repo_map("owner/demo", repo)
    assert "README.md" in rm.key_files
    assert "pyproject.toml" in rm.key_files
    assert rm.languages.get("py", 0) >= 6


def test_license_detection(tmp_path):
    mit = make_repo(tmp_path, "MIT License\n\nCopyright (c) 2026")
    assert detect_license(mit) == LicenseKind.MIT
    apache = make_repo(tmp_path / "a", "Apache License\nVersion 2.0")
    assert detect_license(apache) == LicenseKind.APACHE_2_0
    gpl = make_repo(tmp_path / "g", "GNU GENERAL PUBLIC LICENSE\nVersion 3")
    assert detect_license(gpl) == LicenseKind.GPL_3
    none_repo = make_repo(tmp_path / "n", None)
    assert detect_license(none_repo) == LicenseKind.UNKNOWN


def test_strip_reasoning_variants():
    assert strip_reasoning_content("<think>abc</think>answer")[0] == "answer"
    assert strip_reasoning_content("<THINK>x</think>y")[0] == "y"
    assert strip_reasoning_content("<thought>t</thought>z")[0] == "z"
    text = "<think>one</think><think>two</think>final"
    assert strip_reasoning_content(text)[0] == "final"