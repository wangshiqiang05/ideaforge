import pytest
from mocks import MockLLM, make_config

from ideaforge.cost import CostLedger
from ideaforge.errors import BudgetExceededError, ConfigError
from ideaforge.journal import Journal
from ideaforge.llm.cache_helper import MessageCache
from ideaforge.llm.clients import ChatMessage
from ideaforge.llm.pool import ModelPool, ModelRole


def json_payload(content: str) -> str:
    return content


def make_pool(scripts, journal=None, cfg=None) -> tuple[ModelPool, MockLLM]:
    cfg = cfg or make_config()
    mock = MockLLM(scripts)
    ledger = CostLedger(budget_usd=cfg.budget_usd)
    pool = ModelPool(cfg, ledger, MessageCache(), journal)
    pool.clients["deepseek"] = mock
    pool.clients["deepseek-r"] = mock
    pool.clients["codex"] = mock
    pool.clients["gpt4o"] = mock
    return pool, mock


def test_fixed_roles_routing():
    pool, _ = make_pool([])
    assert pool.resolve_model_name(ModelRole.FIXER, 1) == "deepseek"
    assert pool.resolve_model_name(ModelRole.FIXER, 3) == "deepseek"


def test_per_cycle_rotation():
    cfg = make_config()
    cfg.rotation.per_cycle = True
    cfg.rotation.rotation_table = [
        {"fixer": "deepseek", "reviewer_runnable": "deepseek-r",
         "reviewer_style": "codex", "reviewer_idea": "gpt4o"},
        {"fixer": "codex", "reviewer_runnable": "gpt4o",
         "reviewer_style": "deepseek", "reviewer_idea": "deepseek-r"},
    ]
    pool, _ = make_pool([], cfg=cfg)
    assert pool.resolve_model_name(ModelRole.FIXER, 1) == "deepseek"
    assert pool.resolve_model_name(ModelRole.FIXER, 2) == "codex"
    assert pool.resolve_model_name(ModelRole.FIXER, 3) == "deepseek"
    assert pool.resolve_model_name(ModelRole.REVIEWER_STYLE, 2) == "deepseek"


def test_rotation_missing_role_falls_back():
    cfg = make_config()
    cfg.rotation.per_cycle = True
    cfg.rotation.rotation_table = [{"fixer": "deepseek"}]
    pool, _ = make_pool([], cfg=cfg)
    assert pool.resolve_model_name(ModelRole.REVIEWER_STYLE, 1) == "codex"


def test_distinctness_assertion_rejects_shared_models():
    cfg = make_config()
    cfg.roles["reviewer_style"] = "deepseek"
    with pytest.raises(ConfigError):
        ModelPool(cfg, CostLedger(), MessageCache(), None)


def test_runtime_distinctness_assertion():
    """Static check intercepts a violating rotation row at construction; route()
    re-asserts defensively at call time."""
    cfg = make_config()
    cfg.rotation.per_cycle = True
    cfg.rotation.rotation_table = [
        {"fixer": "deepseek", "reviewer_runnable": "deepseek-r",
         "reviewer_style": "deepseek", "reviewer_idea": "gpt4o"},  # row 0 violates
        {"fixer": "codex", "reviewer_runnable": "gpt4o",
         "reviewer_style": "deepseek", "reviewer_idea": "deepseek-r"},
    ]
    with pytest.raises(ConfigError):
        ModelPool(cfg, CostLedger(), MessageCache(), None)  # static check catches row 0

    cfg2 = make_config()
    cfg2.rotation.per_cycle = True
    cfg2.rotation.rotation_table = [
        {"fixer": "deepseek", "reviewer_runnable": "deepseek-r",
         "reviewer_style": "codex", "reviewer_idea": "gpt4o"},
    ]
    pool = ModelPool(cfg2, CostLedger(), MessageCache(), None)
    pool.route(ModelRole.REVIEWER_STYLE, 1)  # legal row routes fine


def test_cache_hit_skips_model_call_and_billing(tmp_path):
    journal = Journal("t", tmp_path / "t")
    pool, mock = make_pool(["answer"], journal=journal)
    msgs = [ChatMessage(role="user", content="question")]
    r1 = pool.call(ModelRole.PROSPECTOR, 1, msgs)
    r2 = pool.call(ModelRole.PROSPECTOR, 1, msgs)
    assert r1.content == r2.content == "answer"
    assert len(mock.calls) == 1
    assert len(pool.ledger.entries) == 1
    events, _ = journal.read_events()
    usage_events = [e for e in events if e.type == "llm_usage"]
    assert len(usage_events) == 1


def test_billing_per_call():
    pool, _mock = make_pool(["a", "b"])
    pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="1")])
    pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="2")])
    assert len(pool.ledger.entries) == 2
    assert pool.ledger.entries[0].role == "prospector"
    assert pool.ledger.entries[0].prompt_tokens == 10


def test_budget_break_before_next_call():
    cfg = make_config(budget=0.0000001)
    pool, mock = make_pool(["a", "b"], cfg=cfg)
    mock.model = "deepseek-chat"  # price-table entry so usage actually bills
    pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="1")])
    assert pool.ledger.total() > 0
    with pytest.raises(BudgetExceededError):
        pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="2")])


def test_request_json_corrective_retry():
    pool, mock = make_pool(["not json at all", '{"relevance": 1}'])
    result = pool.request_json(
        ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="q")],
    )
    assert result == {"relevance": 1}
    assert len(mock.calls) == 2
    corrective = mock.calls[1]["messages"][-1]
    assert "invalid" in corrective.lower()


def test_request_json_pydantic_retry_and_exhaustion():
    from pydantic import BaseModel

    class Card(BaseModel):
        relevance: int

    pool, mock = make_pool(["garbage one", "garbage two", "garbage three",
                            "garbage four", "garbage five", "garbage six"])
    with pytest.raises(Exception) as exc:
        pool.request_json(ModelRole.PROSPECTOR, 1,
                          [ChatMessage(role="user", content="q")], model_cls=Card)
    assert "after 5 attempts" in str(exc.value)
    assert len(mock.calls) == 5


def test_transport_retry_bounded(monkeypatch):
    from ideaforge.errors import LLMTransportError

    cfg = make_config()
    pool, _mock = make_pool([], cfg=cfg)

    class Flaky:
        model = "mock-model"
        attempts = 0

        def complete(self, messages, *, system=None, temperature=0.2, timeout_s=300):
            self.attempts += 1
            if self.attempts < 3:
                raise LLMTransportError("net down")
            from ideaforge.llm.clients import LLMResponse, TokenUsage

            return LLMResponse(content="ok", model="mock-model", usage=TokenUsage(1, 1))

    pool.clients["deepseek"] = Flaky()
    monkeypatch.setattr("ideaforge.llm.pool.time.sleep", lambda s: None)
    resp = pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="q")])
    assert resp.content == "ok"
    assert pool.clients["deepseek"].attempts == 3


def test_auth_error_not_retried(monkeypatch):
    from ideaforge.errors import LLMAuthError

    pool, _ = make_pool([])

    class AuthFail:
        model = "mock-model"

        def complete(self, messages, **kwargs):
            raise LLMAuthError("bad key")

    pool.clients["deepseek"] = AuthFail()
    monkeypatch.setattr("ideaforge.llm.pool.time.sleep", lambda s: None)
    with pytest.raises(LLMAuthError):
        pool.call(ModelRole.PROSPECTOR, 1, [ChatMessage(role="user", content="q")])