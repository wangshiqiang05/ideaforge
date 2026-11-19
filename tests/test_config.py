import pytest

from ideaforge.config import (
    ForgeConfig,
    ModelSpec,
    ensure_valid,
    load_config,
    validate_config,
)
from ideaforge.errors import ConfigError


def make_valid() -> ForgeConfig:
    cfg = ForgeConfig()
    cfg.models = {
        "deepseek": ModelSpec(model="deepseek-chat"),
        "deepseek-r": ModelSpec(model="deepseek-reasoner"),
        "codex": ModelSpec(model="codex", client="codex_cli"),
        "gpt4o": ModelSpec(model="gpt-4o"),
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
    return cfg


def test_valid_config_passes():
    errors, _warnings = validate_config(make_valid())
    assert errors == []
    ensure_valid(make_valid())


def test_empty_models_pool_reported():
    cfg = make_valid()
    cfg.models = {}
    errors, _ = validate_config(cfg)
    assert any("models pool is empty" in e for e in errors)


def test_missing_required_role_reported():
    cfg = make_valid()
    del cfg.roles["fixer"]
    errors, _ = validate_config(cfg)
    assert any("role 'fixer' has no model assigned" in e for e in errors)


def test_missing_model_reference_reported():
    cfg = make_valid()
    cfg.roles["judge_final"] = "nonexistent"
    errors, _ = validate_config(cfg)
    assert any("undefined model 'nonexistent'" in e for e in errors)


def test_non_positive_budget_reported():
    cfg = make_valid()
    cfg.budget_usd = 0
    errors, _ = validate_config(cfg)
    assert any("budget_usd must be > 0" in e for e in errors)


def test_errors_collected_item_by_item():
    cfg = make_valid()
    cfg.models = {}
    cfg.budget_usd = -1
    with pytest.raises(ConfigError) as exc:
        ensure_valid(cfg)
    assert "models pool is empty" in str(exc.value)
    assert "budget_usd must be > 0" in str(exc.value)


def test_expensive_model_misassignment_warns():
    cfg = make_valid()
    cfg.roles["fixer"] = "gpt4o"
    _errors, warnings = validate_config(cfg)
    assert any("expensive model 'gpt4o'" in w for w in warnings)


def test_identical_fixer_and_reviewer_rejected():
    cfg = make_valid()
    cfg.roles["reviewer_style"] = "deepseek"
    errors, _ = validate_config(cfg)
    assert any("distinct models" in e for e in errors)


def test_per_cycle_rotation_validation():
    cfg = make_valid()
    cfg.rotation.per_cycle = True
    cfg.rotation.rotation_table = [
        {"fixer": "deepseek", "reviewer_runnable": "deepseek-r",
         "reviewer_style": "codex", "reviewer_idea": "gpt4o"},
        {"fixer": "codex", "reviewer_runnable": "gpt4o",
         "reviewer_style": "deepseek", "reviewer_idea": "deepseek-r"},
    ]
    errors, _ = validate_config(cfg)
    assert errors == []

    cfg.rotation.rotation_table[1]["fixer"] = "codex"
    cfg.rotation.rotation_table[1]["reviewer_style"] = "codex"
    errors, _ = validate_config(cfg)
    assert any("row 1" in e and "distinct" in e for e in errors)


def test_load_example_yaml():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg = load_config(path)
    errors, _ = validate_config(cfg)
    assert errors == []
    assert cfg.roles["fixer"] == "deepseek"