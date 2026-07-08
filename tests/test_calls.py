"""Tests for the calls-table end-of-call write policy (twilio vs stream race)."""

import asyncio
import os
import tempfile

from gradphone import tenants


def _fresh_db(monkeypatch):
    path = os.path.join(tempfile.mkdtemp(prefix="gradphone-calls-"), "calls.db")
    monkeypatch.setenv("GRADPHONE_DB", path)


async def _start(room="room1"):
    await tenants.init_db()
    await tenants.record_call_start(
        room=room, tenant_id=1, twilio_call_sid="CA123",
        destination="+15551234567", task="ask hours", language="en",
        business_name="Demo Cafe",
    )


def test_stream_result_overwrites_twilio_placeholder(monkeypatch):
    """Twilio's completed-callback lands first with 'unclear'; the stream's
    real answer must still win, and the Twilio metadata must survive."""
    _fresh_db(monkeypatch)

    async def run():
        await _start()
        await tenants.record_call_end(
            room="room1", status="unclear",
            twilio_call_status="completed", answered_by="human",
            duration_seconds=42.0, source="twilio",
        )
        await tenants.record_call_end(
            room="room1", status="answered", answer="Open until 6pm.",
            confidence="high", duration_seconds=44.5, source="stream",
        )
        return await tenants.get_call("room1")

    row = asyncio.run(run())
    assert row["status"] == "answered"
    assert row["answer"] == "Open until 6pm."
    # Stream side didn't know these; the placeholder's values are preserved.
    assert row["twilio_call_status"] == "completed"
    assert row["answered_by"] == "human"


def test_twilio_does_not_clobber_stream_result(monkeypatch):
    """Reverse order: the stream wrote the real result first; a late Twilio
    callback must not overwrite it."""
    _fresh_db(monkeypatch)

    async def run():
        await _start()
        await tenants.record_call_end(
            room="room1", status="answered", answer="Yes, gluten-free options.",
            confidence="high", duration_seconds=30.0, source="stream",
        )
        await tenants.record_call_end(
            room="room1", status="unclear",
            twilio_call_status="completed", duration_seconds=31.0, source="twilio",
        )
        return await tenants.get_call("room1")

    row = asyncio.run(run())
    assert row["status"] == "answered"
    assert row["answer"] == "Yes, gluten-free options."


def test_no_answer_outcome_is_final(monkeypatch):
    """busy/no-answer calls never connect a stream; the Twilio outcome stands
    and a stray stream write must not resurrect them."""
    _fresh_db(monkeypatch)

    async def run():
        await _start()
        await tenants.record_call_end(
            room="room1", status="no_answer",
            twilio_call_status="no-answer", duration_seconds=0.0, source="twilio",
        )
        # Hypothetical stray stream write (shouldn't happen, must not clobber).
        await tenants.record_call_end(
            room="room1", status="answered", answer="ghost", source="stream",
        )
        return await tenants.get_call("room1")

    row = asyncio.run(run())
    assert row["status"] == "no_answer"
    assert row["answer"] == ""
