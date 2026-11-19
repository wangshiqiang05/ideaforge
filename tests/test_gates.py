from mocks import (
    ExecResult,
    FakeContainer,
    FakeSandboxManager,
    make_config,
    make_journal,
)

from ideaforge.errors import ExecTimeoutError
from ideaforge.forge.gates import ToolGate
from ideaforge.state import GateStage


def make_gate(exec_script=None):
    journal = make_journal(__import__("pathlib").Path(__file__).parent / "ws-gates", "t-gates")
    sandbox = FakeSandboxManager(exec_script=exec_script, journal=journal)
    gate = ToolGate(sandbox, journal, make_config())
    return gate, sandbox, journal


def test_three_stages_run_in_order_on_success():
    gate, sandbox, _ = make_gate()
    report = gate.run("t1", FakeContainer())
    assert report.passed is True
    assert report.failed_stage is None
    assert len(sandbox.exec_calls) == 3
    assert any("compileall" in c for c in sandbox.exec_calls[0])
    assert any("ruff" in c for c in sandbox.exec_calls[1])
    assert any("pytest" in c for c in sandbox.exec_calls[2])


def test_syntax_failure_short_circuits_zero_llm():
    gate, sandbox, journal = make_gate(exec_script=[
        ExecResult(success=False, exit_code=1, output="SyntaxError: bad line 3"),
    ])
    report = gate.run("t1", FakeContainer())
    assert report.passed is False
    assert report.failed_stage == GateStage.SYNTAX
    assert "SyntaxError" in report.raw_output
    assert len(sandbox.exec_calls) == 1  # lint/test never ran
    events, _ = journal.read_events()
    assert any(e.type == "tool_gate_result" for e in events)


def test_lint_failure_short_circuits():
    gate, sandbox, _ = make_gate(exec_script=[
        ExecResult(success=True, exit_code=0, output=""),
        ExecResult(success=False, exit_code=1, output="F401 unused import"),
    ])
    report = gate.run("t1", FakeContainer())
    assert report.failed_stage == GateStage.LINT
    assert len(sandbox.exec_calls) == 2


def test_pytest_failure_keeps_verbatim_output():
    gate, _, _ = make_gate(exec_script=[
        ExecResult(success=True, exit_code=0, output=""),
        ExecResult(success=True, exit_code=0, output=""),
        ExecResult(success=False, exit_code=1, output="1 failed, 2 passed in 0.1s"),
    ])
    report = gate.run("t1", FakeContainer())
    assert report.failed_stage == GateStage.TEST
    assert "1 failed, 2 passed" in report.raw_output


def test_timeout_output_routed_to_report():
    timeout = ExecTimeoutError("pytest timed out after 900s")
    gate, sandbox, _ = make_gate(exec_script=[timeout, timeout, timeout])
    report = gate.run("t1", FakeContainer())
    assert report.passed is False
    assert report.failed_stage == GateStage.SYNTAX
    assert "timed out" in report.raw_output
    assert len(sandbox.exec_calls) == 3  # 1 + 2 retries


def test_tool_gate_never_calls_models():
    gate, _sandbox, _ = make_gate()
    # No LLM client exists at all; the run must complete without touching any.
    report = gate.run("t1", FakeContainer())
    assert report.passed