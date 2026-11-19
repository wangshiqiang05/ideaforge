import pytest

from ideaforge.errors import IllegalTransitionError
from ideaforge.journal import Journal
from ideaforge.state import Task, TaskStateMachine, TaskStatus


def make(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    task = Task(task_id="t1", idea="idea")
    return TaskStateMachine(task, journal), journal


def test_all_legal_transitions(tmp_path):
    fsm, _ = make(tmp_path)
    fsm.transition(TaskStatus.AWAITING_APPROVAL, "plan ready")
    fsm.transition(TaskStatus.RUNNING, "approved")
    fsm.transition(TaskStatus.AWAITING_BUDGET_DECISION, "budget")
    fsm.transition(TaskStatus.RUNNING, "raised")
    fsm.transition(TaskStatus.COMPLETED, "converged")


def test_degraded_only_from_running(tmp_path):
    fsm, _ = make(tmp_path)
    fsm.transition(TaskStatus.DEGRADED, "cycles exhausted")
    with pytest.raises(IllegalTransitionError):
        fsm.transition(TaskStatus.RUNNING, "cannot revive")


def test_awaiting_approval_cannot_complete(tmp_path):
    fsm, journal = make(tmp_path)
    fsm.transition(TaskStatus.AWAITING_APPROVAL, "plan ready")
    with pytest.raises(IllegalTransitionError):
        fsm.transition(TaskStatus.COMPLETED, "illegal shortcut")
    events, _ = journal.read_events()
    rejected = [e for e in events if e.type == "state_change" and e.payload.get("rejected")]
    assert rejected, "illegal transition must be journaled"
    assert rejected[0].payload["from"] == "awaiting_approval"
    assert rejected[0].payload["to"] == "completed"


def test_awaiting_budget_decision_only_running_or_failed(tmp_path):
    fsm, _ = make(tmp_path)
    fsm.transition(TaskStatus.AWAITING_BUDGET_DECISION, "budget")
    with pytest.raises(IllegalTransitionError):
        fsm.transition(TaskStatus.DEGRADED, "illegal")
    fsm.transition(TaskStatus.FAILED, "user aborted")
    assert fsm.task.status == TaskStatus.FAILED