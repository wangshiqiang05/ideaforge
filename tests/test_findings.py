
from ideaforge.forge.findings import (
    OscillationDetector,
    aggregate,
    findings_hash,
    tool_findings,
)
from ideaforge.state import (
    Finding,
    FindingCategory,
    GateStage,
    ReviewerKind,
    Severity,
    ToolReport,
    Verdict,
    VerdictKind,
)


def make_verdict(kind, findings, cycle=1):
    return Verdict(verdict=kind, reviewer=kind and ReviewerKind.R1_RUNNABLE,
                   confidence=0.9, findings=findings, cycle=cycle)


def test_tool_findings_empty_when_passed():
    assert tool_findings(ToolReport(passed=True, raw_output="")) == []


def test_tool_findings_carry_verbatim_output():
    report = ToolReport(passed=False, failed_stage=GateStage.TEST,
                        raw_output="FAILED tests/test_x.py::test_a - assert 1 == 2")
    findings = tool_findings(report)
    assert len(findings) == 1
    assert findings[0].category == FindingCategory.RUNNABLE
    assert "assert 1 == 2" in findings[0].description
    assert findings[0].source.startswith("toolgate:")


def test_aggregate_merges_and_dedupes():
    report = ToolReport(passed=True, raw_output="")
    f1 = Finding(file="src/a.py", line=1, severity=Severity.MAJOR,
                 category=FindingCategory.STYLE, description="dup")
    f2 = Finding(file="src/a.py", line=1, severity=Severity.MAJOR,
                 category=FindingCategory.STYLE, description="dup")
    v1 = Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R1_RUNNABLE,
                 confidence=0.8, findings=[f1], cycle=1)
    v2 = Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R2_STYLE,
                 confidence=0.8, findings=[f2], cycle=1)
    merged = aggregate(report, [v1, v2])
    assert len(merged) == 1
    assert merged[0].source == "R1_runnable"


def test_aggregate_keeps_different_findings():
    report = ToolReport(passed=True, raw_output="")
    f1 = Finding(file="src/a.py", line=1, severity=Severity.MAJOR,
                 category=FindingCategory.STYLE, description="one")
    f2 = Finding(file="src/a.py", line=2, severity=Severity.MINOR,
                 category=FindingCategory.STYLE, description="two")
    v = Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R1_RUNNABLE,
                confidence=0.8, findings=[f1, f2], cycle=1)
    merged = aggregate(report, [v])
    assert len(merged) == 2


def test_critical_findings_detectable():
    f = Finding(file="src/x.py", line=None, severity=Severity.CRITICAL,
                category=FindingCategory.RUNNABLE, description="security hole")
    v = Verdict(verdict=VerdictKind.FAIL, reviewer=ReviewerKind.R1_RUNNABLE,
                confidence=0.9, findings=[f], cycle=1)
    merged = aggregate(ToolReport(passed=True, raw_output=""), [v])
    assert any(x.severity == Severity.CRITICAL for x in merged)


def test_findings_hash_stable():
    f1 = Finding(file="a.py", line=1, severity=Severity.MAJOR,
                 category=FindingCategory.STYLE, description="d")
    f2 = f1.model_copy()
    assert findings_hash([f1]) == findings_hash([f2])
    f3 = Finding(file="a.py", line=2, severity=Severity.MAJOR,
                 category=FindingCategory.STYLE, description="d")
    assert findings_hash([f1]) != findings_hash([f3])


def test_oscillation_n_vs_n_minus_2():
    detector = OscillationDetector()
    detector.record(1, "hash-a")
    detector.record(2, "hash-b")
    assert detector.check(3, "hash-a") is True
    assert detector.check(3, "hash-other") is False
    assert detector.check(2, "hash-a") is False  # no N-2 for cycle 2


def test_oscillation_escalate_once_then_terminate():
    detector = OscillationDetector()
    detector.record(1, "h")
    detector.record(2, "h")
    assert detector.check(3, "h") is True
    assert detector.should_terminate(True) is False  # first oscillation escalates
    detector.record(3, "h")
    assert detector.check(4, "h") is True
    assert detector.should_terminate(True) is True   # second one terminates