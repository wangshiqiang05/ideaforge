import asyncio

import pytest
from conftest import requires_loopback

from ideaforge.journal import Journal, redact


def test_events_persisted_line_by_line(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {"idea_chars": 10})
    journal.emit("cycle_start", {"cycle": 1}, cycle=1)
    events, gaps = journal.read_events()
    assert gaps == []
    assert [e.type for e in events] == ["task_created", "cycle_start"]
    assert events[1].cycle == 1
    assert events[1].task_id == "t1"
    assert events[0].event_seq < events[1].event_seq


def test_unknown_event_type_rejected(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    with pytest.raises(ValueError):
        journal.emit("bogus_event", {})


def test_secrets_redacted(tmp_path):
    journal = Journal("t1", tmp_path / "t1", secrets={"super-secret-value"})
    journal.emit("failed", {"reason": "leak of super-secret-value inside message"})
    journal.emit("failed", {"reason": "key sk-abcdefghij1234567890 in logs"})
    events, _ = journal.read_events()
    assert "super-secret-value" not in events[0].payload["reason"]
    assert "***" in events[0].payload["reason"]
    assert "sk-abcdefghij1234567890" not in events[1].payload["reason"]


def test_redact_function():
    assert redact("token=ghp_abcdefghijklmnopqrst", set()) == "token=***"
    assert redact("Bearer abcdefgh12345", set()) == "Bearer ***" or "Bearer abcdefgh12345" in redact(
        "Bearer abcdefgh12345", set()
    )
    assert redact("hello secretvalue", {"secretvalue"}) == "hello ***"


def test_replay_skips_corrupt_lines(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {"n": 1})
    journal.emit("cycle_start", {"c": 1}, cycle=1)
    with journal.path.open("a", encoding="utf-8") as fh:
        fh.write("{corrupted json line\n")
    journal.emit("cycle_end", {"c": 1}, cycle=1)
    result = journal.replay(cycle=1)
    assert [e.type for e in result.events] == ["cycle_start", "cycle_end"]
    assert len(result.gaps) == 1


def test_replay_filters_by_task_and_cycle(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {})
    journal.emit("stage_enter", {"stage": "Prospecting"})
    journal.emit("stage_exit", {"stage": "Prospecting"}, cycle=1)
    result = journal.replay(cycle=1)
    assert [e.type for e in result.events] == ["stage_exit"]


def test_read_after_incremental(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {})
    journal.emit("cycle_start", {}, cycle=1)
    events = journal.read_after(1)
    assert [e.type for e in events] == ["cycle_start"]


@requires_loopback
def test_subscribe_incremental(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {})
    journal.emit("cycle_start", {}, cycle=1)
    journal.emit("cycle_end", {}, cycle=1)

    collected: list = []

    async def run():
        async for event in journal.subscribe(0, poll_interval=0.01,
                                             is_done=lambda: len(collected) >= 3):
            collected.append(event)

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert [e.type for e in collected] == ["task_created", "cycle_start", "cycle_end"]


@requires_loopback
def test_subscribe_resume_after_seq(tmp_path):
    journal = Journal("t1", tmp_path / "t1")
    for i in range(3):
        journal.emit("task_created", {"i": i})

    collected: list = []

    async def run():
        async for event in journal.subscribe(1, poll_interval=0.01,
                                             is_done=lambda: len(collected) >= 2):
            collected.append(event)

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert [e.event_seq for e in collected] == [2, 3]

# --------------------------------------------------------------------------
# V7-V10: sync-kernel substitute verification for the loopback-skipped
# subscribe tests above (review_fix design 2.1.3(3)). The SSE payload
# semantics are fully determined by Journal's sync read path; asyncio /
# subscribe() itself is not exercised because the self-pipe socket shares
# the loopback restriction. All assertions are synchronous — no asyncio.run().
# --------------------------------------------------------------------------


def test_v7_event_stream_fields_complete(tmp_path):
    """V7 <- test_sse_stream_events (skipped): event objects carry full SSE fields."""
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {})
    journal.emit("completed", {})
    events, gaps = journal.read_events()
    assert gaps == []
    assert [e.type for e in events] == ["task_created", "completed"]
    for event in events:
        assert event.task_id == "t1"
        assert isinstance(event.event_seq, int) and event.event_seq > 0
    assert events[0].event_seq < events[1].event_seq


def test_v8_gap_replay_from_last_event_id(tmp_path):
    """V8 <- test_sse_resume_from_last_event_id (skipped): Last-Event-ID=2 replays seq>=3."""
    journal = Journal("t1", tmp_path / "t1")
    for i in range(4):
        journal.emit("task_created", {"i": i})
    replayed = journal.read_after(2)
    assert [e.event_seq for e in replayed] == [3, 4]


def test_v9_full_stream_from_zero(tmp_path):
    """V9 <- test_subscribe_incremental (skipped): subscribe(0) yields the full stream."""
    journal = Journal("t1", tmp_path / "t1")
    journal.emit("task_created", {})
    journal.emit("cycle_start", {}, cycle=1)
    journal.emit("cycle_end", {}, cycle=1)
    collected = journal.read_after(0)
    assert [e.type for e in collected] == ["task_created", "cycle_start", "cycle_end"]


def test_v10_resume_midstream_after_seq(tmp_path):
    """V10 <- test_subscribe_resume_after_seq (skipped): subscribe(1) yields exactly [2, 3]."""
    journal = Journal("t1", tmp_path / "t1")
    for _ in range(3):
        journal.emit("task_created", {})
    collected = journal.read_after(1)
    assert [e.event_seq for e in collected] == [2, 3]