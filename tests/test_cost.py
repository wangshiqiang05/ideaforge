import pytest

from ideaforge.cost import CostLedger
from ideaforge.llm.clients import TokenUsage


def test_multi_role_accumulation():
    ledger = CostLedger(budget_usd=100.0)
    ledger.add("fixer", "deepseek-chat", TokenUsage(1_000_000, 100_000))
    ledger.add("reviewer_idea", "gpt-4o", TokenUsage(500_000, 50_000))
    # deepseek: 1M*0.27 + 0.1M*1.10 = 0.27+0.11=0.38 ; gpt4o: 0.5M*2.5+0.05M*10=1.25+0.5=1.75
    assert ledger.total() == pytest.approx(0.38 + 1.75)
    by_role = ledger.by_role()
    assert by_role["fixer"] == pytest.approx(0.38)
    assert by_role["reviewer_idea"] == pytest.approx(1.75)


def test_ledger_only_grows():
    ledger = CostLedger()
    before = len(ledger.entries)
    ledger.add("fixer", "deepseek-chat", TokenUsage(10, 5))
    assert len(ledger.entries) == before + 1


def test_missing_price_bills_zero_with_flag():
    ledger = CostLedger(budget_usd=100.0)
    entry = ledger.add("fixer", "unknown-model", TokenUsage(1000, 1000))
    assert entry.usd == 0.0
    assert entry.billed is False


def test_single_call_alert():
    alerts = []
    ledger = CostLedger(budget_usd=1.0)
    ledger.on_alert = lambda entry: alerts.append(entry)
    # 0.45 USD >= 30% of 1.0 budget -> alert
    entry = ledger.add("reviewer_idea", "gpt-4o", TokenUsage(150_000, 7_500))
    assert entry.alert is True
    assert len(alerts) == 1
    small = ledger.add("fixer", "deepseek-chat", TokenUsage(1_000, 100))
    assert small.alert is False


def test_exceeds_threshold():
    ledger = CostLedger(budget_usd=0.5)
    assert not ledger.exceeds()
    ledger.add("fixer", "gpt-4o", TokenUsage(200_000, 0))  # 0.5 USD
    assert ledger.exceeds()
    assert ledger.exceeds(10.0) is False


def test_snapshot_shape():
    ledger = CostLedger(budget_usd=10.0)
    ledger.add("fixer", "deepseek-chat", TokenUsage(10, 5))
    snap = ledger.snapshot()
    assert snap.total_usd >= 0
    assert "fixer" in snap.by_role