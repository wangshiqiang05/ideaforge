"""Shared test doubles: MockLLM (scripted responses), FakeSandbox/FakeDocker,
and pipeline factories for loop-level regression tests (design 2.8.3.1)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from docker.errors import ImageNotFound

from ideaforge.config import ForgeConfig, ModelSpec
from ideaforge.cost import CostLedger
from ideaforge.errors import SandboxError
from ideaforge.journal import Journal
from ideaforge.llm.cache_helper import MessageCache
from ideaforge.llm.clients import ChatMessage, LLMResponse, TokenUsage
from ideaforge.llm.pool import ModelPool
from ideaforge.sandbox.manager import ContainerSpec, ExecResult, SandboxManager
from ideaforge.state import (
    Finding,
    PipelineState,
    ReviewerKind,
    Severity,
    Task,
    TaskConfig,
    Verdict,
    VerdictKind,
)


def make_config(max_cycles: int = 6, budget: float = 100.0) -> ForgeConfig:
    cfg = ForgeConfig()
    cfg.models = {
        "deepseek": ModelSpec(model="deepseek-chat", base_url="https://api.deepseek.com/v1",
                              api_key_env="DEEPSEEK_API_KEY"),
        "deepseek-r": ModelSpec(model="deepseek-reasoner", base_url="https://api.deepseek.com/v1",
                                api_key_env="DEEPSEEK_API_KEY"),
        "gpt4o": ModelSpec(model="gpt-4o", api_key_env="OPENAI_API_KEY"),
        "codex": ModelSpec(model="codex", client="codex_cli", cmd="codex",
                           api_key_env="OPENAI_API_KEY"),
    }
    cfg.roles = {
        "fixer": "deepseek",
        "reviewer_runnable": "deepseek-r",
        "reviewer_style": "codex",
        "reviewer_idea": "gpt4o",
        "judge_final": "gpt4o",
        "prospector": "deepseek",
        "architect": "deepseek",
        "synthesizer": "deepseek",
    }
    cfg.max_cycles = max_cycles
    cfg.budget_usd = budget
    return cfg


def make_state(tmp_path: Path, idea: str = "test idea", max_cycles: int = 6) -> PipelineState:
    task = Task(
        task_id="t-test",
        idea=idea,
        config=TaskConfig(max_cycles=max_cycles, budget_usd=100.0),
    )
    return PipelineState(task=task)


class MockLLM:
    """Scripted LLMClient: returns preset payloads in order, records calls."""

    def __init__(self, scripts: list[str]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.model = "mock-model"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        timeout_s: int = 300,
    ) -> LLMResponse:
        self.calls.append({
            "messages": [m.content for m in messages],
            "system": system,
        })
        if not self.scripts:
            raise AssertionError("MockLLM exhausted: unexpected call")
        payload = self.scripts.pop(0)
        return LLMResponse(content=payload, model=self.model, usage=TokenUsage(10, 5))


def verdict_json(passed: bool, findings: list[dict] | None = None, reviewer="R1_runnable") -> str:
    body = {
        "verdict": "PASS" if passed else "FAIL",
        "confidence": 0.9,
        "findings": findings or [],
    }
    return json.dumps(body)


def finding(file="src/x.py", line=1, severity="major", category="style",
            description="something is off", item_id=None) -> dict:
    return {
        "file": file,
        "line": line,
        "severity": severity,
        "category": category,
        "description": description,
        "suggested_fix": "rewrite it",
        "item_id": item_id,
    }


class NoCache(MessageCache):
    """Cache double that always misses - keeps scripted call sequencing exact."""

    def get(self, key: str):
        self.misses += 1

    def put(self, key: str, value) -> None:
        pass


@dataclass
class FakeContainer:
    name: str = "fake"
    committed: list = field(default_factory=list)
    execs: list = field(default_factory=list)


class FakeDocker:
    """Docker SDK double: images.get raises unless seeded; containers.run fakes."""

    def __init__(self, images: set[str] | None = None, exec_script: list | None = None) -> None:
        self._images = set(images or [])
        self.exec_script = list(exec_script or [])
        self.containers = self
        self.created: list[ContainerSpec] = []

    def images(self):
        return self

    def get(self, tag: str):
        if tag in self._images:
            return object()
        raise ImageNotFound(f"image {tag} not found")

    def run(self, image, command=None, detach=False, name=None, network=None,
            mem_limit=None, nano_cpus=None, volumes=None, working_dir=None, **kwargs):
        if network == "host":
            raise SandboxError("host network is forbidden by the sandbox red lines (spec 4.3.1)")
        spec = ContainerSpec(
            image=image, name=name or "c", workdir=working_dir or "/forge",
            network=network, mem_limit=mem_limit or "4g",
            cpu_count=(nano_cpus or 2_000_000_000) / 1e9,
        )
        self.created.append(spec)
        return FakeContainer(name=name or "c")

    def commit(self, repository, tag):
        return object()


class FakeSandboxManager(SandboxManager):
    """Scripted sandbox: each _exec_once pops one scripted result; the parent
    exec() retry/denylist semantics stay intact."""

    def __init__(self, exec_script: list | None = None, images: set[str] | None = None, journal=None):
        super().__init__(docker_client=FakeDocker(images=images), journal=journal)
        self.exec_script = list(exec_script or [])
        self.exec_calls: list[list[str]] = []
        self.committed: list[str] = []
        self.destroyed = 0

    def _exec_once(self, handle, cmd, timeout_s, workdir) -> ExecResult:
        self.exec_calls.append([str(c) for c in cmd])
        if not self.exec_script:
            return ExecResult(success=True, exit_code=0, output="ok")
        item = self.exec_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def commit_image(self, handle, tag):
        self.committed.append(tag)
        return type("ImageRef", (), {"tag": f"{tag}:latest"})()

    def ensure_image(self, repo_full_name: str):
        tag = repo_full_name.replace("/", "_")
        if tag in self._docker._images:
            return type("ImageRef", (), {"tag": f"{tag}:latest"})()
        return None

    def destroy(self, handle):
        self.destroyed += 1


def make_journal(tmp_path: Path, task_id: str = "t-test") -> Journal:
    return Journal(task_id, tmp_path / task_id)


def make_git_repo(path: Path, files: dict[str, str]) -> Path:
    """Create a real local git repo with the given files (for clone tests)."""
    import git

    path.mkdir(parents=True, exist_ok=True)
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
    return path


class FakeHTTP:
    """GitHub API double keyed on the `q` query param."""

    def __init__(self, by_query: dict[str, list[dict]], fail_queries: set[str] | None = None):
        self.by_query = by_query
        self.fail_queries = fail_queries or set()
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        q = (params or {}).get("q", "")

        class Resp:
            pass

        resp = Resp()
        if q in self.fail_queries:
            resp.status_code = 500
            resp.json = lambda: {"message": "boom"}
            return resp
        items = self.by_query.get(q, [])
        resp.status_code = 200
        resp.json = lambda: {"items": items}
        return resp


def prospect(full_name: str, clone_url: str = "", score: int = 4, license_name="MIT"):
    from ideaforge.state import LicenseKind, Prospect, ProspectStatus, ScoreCard

    try:
        lic = LicenseKind(license_name)
    except ValueError:
        lic = LicenseKind.UNKNOWN
    return Prospect(
        repo_full_name=full_name,
        clone_url=clone_url,
        score_card=ScoreCard(relevance=score, modifiability=score, quality=score,
                             license_compat=score, license=lic),
        status=ProspectStatus.SCORED,
    )


def make_pool(scripts=None, cfg: ForgeConfig | None = None, journal=None,
              client_overrides: dict | None = None) -> tuple[ModelPool, MockLLM]:
    cfg = cfg or make_config()
    mock = MockLLM(list(scripts or []))
    mock.model = "deepseek-chat"  # priced model so usage actually bills
    ledger = CostLedger(budget_usd=cfg.budget_usd)
    pool = ModelPool(cfg, ledger, MessageCache(), journal)
    for name in ("deepseek", "deepseek-r", "codex", "gpt4o"):
        pool.clients[name] = mock  # tests always talk to the scripted mock
    for name, client in (client_overrides or {}).items():
        pool.clients[name] = client
    return pool, mock


def pass_verdict(reviewer: ReviewerKind, cycle: int) -> Verdict:
    return Verdict(verdict=VerdictKind.PASS, reviewer=reviewer, confidence=0.95, findings=[], cycle=cycle)


def fail_verdict(reviewer: ReviewerKind, cycle: int, findings: list[Finding] | None = None) -> Verdict:
    findings = findings or [
        Finding(file="src/x.py", line=1, severity=Severity.MAJOR, category="style",
                description="bad style", suggested_fix="fix it")
    ]
    return Verdict(verdict=VerdictKind.FAIL, reviewer=reviewer, confidence=0.8,
                   findings=findings, cycle=cycle)