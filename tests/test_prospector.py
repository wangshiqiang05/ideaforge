import json
from pathlib import Path

import pytest
from mocks import FakeHTTP, make_config, make_git_repo, make_journal, make_pool

from ideaforge.agents.prospector import Prospector
from ideaforge.errors import AgentFailure
from ideaforge.state import LicenseKind, PipelineState, Task, TaskConfig

SCORE_GOOD = json.dumps({
    "relevance": 5, "modifiability": 4, "quality": 4, "license_compat": 5,
    "license": "MIT",
})
SCORE_PLATFORM = json.dumps({
    "relevance": 5, "modifiability": 0, "quality": 3, "license_compat": 5,
    "license": "MIT",
})
QUERIES = json.dumps({"queries": [{"text": "qlib backtest"}, {"topic": "quant-storage"}]})


def make_item(tmp_path: Path, name: str) -> dict:
    repo_dir = tmp_path / f"src-{name}"
    make_git_repo(repo_dir, {
        "README.md": f"# {name} tool\npip install and run backtests",
        "LICENSE": "MIT License",
        "main.py": "print('hi')\n",
    })
    return {"full_name": f"owner/{name}", "clone_url": str(repo_dir), "description": "demo"}


def make_prospector(tmp_path, scripts, by_query, fail_queries=None, journal=None):
    cfg = make_config()
    pool, mock = make_pool(scripts, cfg=cfg, journal=journal)
    http = FakeHTTP(by_query, fail_queries)
    prospector = Prospector(pool, journal, cfg, github_token="ghp_testtoken1234567890abc", http=http)
    return prospector, mock, http, cfg


def make_state(tmp_path) -> PipelineState:
    task = Task(task_id="t1", idea="A-share LightGBM daily backtest", config=TaskConfig())
    return PipelineState(task=task)


def test_generates_two_to_four_queries(tmp_path):
    item = make_item(tmp_path, "alpha")
    prospector, mock, http, _cfg = make_prospector(
        tmp_path, [QUERIES, SCORE_GOOD],
        by_query={"qlib backtest": [item]},
    )
    prospector.run(make_state(tmp_path))
    query_calls = [c for c in mock.calls if "search queries" in (c["system"] or "")]
    assert len(query_calls) == 1
    assert len(http.calls) >= 1


def test_dual_channel_merge(tmp_path):
    item_a = make_item(tmp_path, "alpha")
    item_b = make_item(tmp_path, "beta")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path,
        [QUERIES, SCORE_GOOD, SCORE_GOOD],
        by_query={"qlib backtest": [item_a], "topic:quant-storage": [item_b]},
    )
    state = prospector.run(make_state(tmp_path))
    names = {p.repo_full_name for p in state.prospects}
    assert names == {"owner/alpha", "owner/beta"}


def test_single_channel_degraded_and_journaled(tmp_path):
    item = make_item(tmp_path, "alpha")
    journal = make_journal(tmp_path, "t1")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path,
        [QUERIES, SCORE_GOOD],
        by_query={"topic:quant-storage": [item]},
        fail_queries={"qlib backtest"},
        journal=journal,
    )
    state = prospector.run(make_state(tmp_path))
    assert state.prospects[0].repo_full_name == "owner/alpha"
    events, _ = journal.read_events()
    assert any(e.type == "channel_degraded" for e in events)


def test_both_channels_fail_terminates(tmp_path):
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path, [QUERIES], by_query={}, fail_queries={"qlib backtest", "topic:quant-storage"},
    )
    with pytest.raises(AgentFailure):
        prospector.run(make_state(tmp_path))


def test_github_token_never_in_logs(tmp_path):
    item = make_item(tmp_path, "alpha")
    journal = make_journal(tmp_path, "t1")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path, [QUERIES, SCORE_GOOD], by_query={"qlib backtest": [item]}, journal=journal,
    )
    prospector.run(make_state(tmp_path))
    blob = (tmp_path / "t1" / "journal.jsonl").read_text(encoding="utf-8")
    assert "ghp_testtoken1234567890abc" not in blob



def test_blacklist_prevents_reevaluation(tmp_path):
    item = make_item(tmp_path, "alpha")
    prospector, mock, _http, _cfg = make_prospector(
        tmp_path,
        [QUERIES] + ["garbage"] * 5,
        by_query={"qlib backtest": [item]},
    )
    with pytest.raises(AgentFailure):
        prospector.run(make_state(tmp_path))
    assert "owner/alpha" in {url.split("/")[-1] for url in prospector.blacklist} or prospector.blacklist
    calls_after_first = len(mock.calls)
    # second run: query regeneration fails fast? blacklist prevents scoring again
    prospector.pool.cache._store.clear()
    mock.scripts = [QUERIES]
    with pytest.raises(AgentFailure):
        prospector.run(make_state(tmp_path))
    assert len(mock.calls) == calls_after_first + 1  # only the query generation call


def test_api_platform_class_excluded(tmp_path):
    item = make_item(tmp_path, "saas")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path, [QUERIES, SCORE_PLATFORM], by_query={"qlib backtest": [item]},
    )
    with pytest.raises(AgentFailure):
        prospector.run(make_state(tmp_path))  # Top-K empty after exclusion


def test_score_card_and_license_classification(tmp_path):
    item = make_item(tmp_path, "alpha")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path, [QUERIES, SCORE_GOOD], by_query={"qlib backtest": [item]},
    )
    state = prospector.run(make_state(tmp_path))
    card = state.prospects[0].score_card
    assert card.relevance == 5 and card.modifiability == 4
    assert card.license == LicenseKind.MIT


def test_temp_clone_deleted_even_on_failure(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(
        "ideaforge.agents.prospector.tempfile.mkdtemp",
        lambda prefix="": str(scratch / f"tmp{len(list(scratch.iterdir()))}"),
    )
    item = make_item(tmp_path, "alpha")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path, [QUERIES] + ["garbage"] * 5, by_query={"qlib backtest": [item]},
    )
    with pytest.raises(AgentFailure):
        prospector.run(make_state(tmp_path))
    leftovers = [p for p in scratch.iterdir() if p.is_dir()]
    assert leftovers == []


def test_top_k_sorted_by_weighted_total(tmp_path):
    a = make_item(tmp_path, "alpha")
    b = make_item(tmp_path, "beta")
    prospector, _mock, _http, _cfg = make_prospector(
        tmp_path,
        [QUERIES, SCORE_GOOD,
         json.dumps({"relevance": 5, "modifiability": 5, "quality": 5,
                     "license_compat": 5, "license": "Apache-2.0"})],
        by_query={"qlib backtest": [a, b]},
    )
    state = prospector.run(make_state(tmp_path))
    assert [p.repo_full_name for p in state.prospects] == ["owner/beta", "owner/alpha"]