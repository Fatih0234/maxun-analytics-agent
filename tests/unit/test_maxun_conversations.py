from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import orjson
import pytest
from analytics_agent.api import maxun_conversations as adapter
from analytics_agent.db.models import Base
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

WORKSPACE_ID = "11111111-1111-4111-8111-111111111111"
SIGNATURE = "a" * 64
TOKEN = "internal-test-token"
TURN_ID = "99999999-9999-4999-8999-999999999999"


class FakeEngine:
    def get_tools(self):
        return []

    async def aclose(self):
        return None


@pytest.fixture
def app_and_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "agent.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def create_schema():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(create_schema())
    monkeypatch.setattr(adapter, "_get_session_factory", lambda: sessions)
    monkeypatch.setenv("MAXUN_ANALYTICS_INTERNAL_TOKEN", TOKEN)
    monkeypatch.delenv("MOCK_LLM", raising=False)

    async def fake_resolve_engine(*args, **kwargs):
        return _resolved_engine()

    monkeypatch.setattr(adapter, "resolve_engine", fake_resolve_engine)
    monkeypatch.setattr(adapter, "build_graph", lambda **kwargs: object())

    async def fake_events(**kwargs) -> AsyncIterator[dict]:
        yield {
            "event": "TOOL_CALL",
            "payload": {
                "tool_name": "execute_sql",
                "tool_input": {"sql": "SELECT COUNT(*) FROM data"},
            },
        }
        yield {
            "event": "SQL",
            "payload": {
                "sql": "SELECT COUNT(*) FROM data",
                "columns": ["count"],
                "rows": [{"count": 2}],
                "truncated": False,
            },
        }
        yield {"event": "TEXT", "payload": {"text": "There are 2 rows."}}
        yield {"event": "COMPLETE", "payload": {"text": "There are 2 rows."}}

    monkeypatch.setattr(adapter, "stream_graph_events", fake_events)

    application = FastAPI()
    application.include_router(adapter.router)
    yield application
    asyncio.run(engine.dispose())


def _resolved_engine():
    return FakeEngine()


async def _request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_internal_boundary_and_completed_turn(app_and_db):
    response = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    assert response.status_code == 503

    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    assert response.status_code == 201
    conversation_id = response.json()["conversation_id"]

    response = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": TURN_ID,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    expected = {
        "status": "completed",
        "answer": "There are 2 rows.",
        "sql": "SELECT COUNT(*) FROM data",
        "columns": ["count"],
        "rows": [{"count": 2}],
        "truncated": False,
    }
    assert response.json() == expected

    replay = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": TURN_ID,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert replay.status_code == 200
    assert replay.json() == expected


@pytest.mark.asyncio
async def test_maxun_turn_uses_history_compactor(app_and_db, monkeypatch):
    captured = {}

    def capture_history(stored, current, compactor=None, max_history_tokens=0):
        captured["compactor"] = compactor
        captured["budget"] = max_history_tokens
        return []

    monkeypatch.setattr(adapter, "build_history", capture_history)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    response = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{created.json()['conversation_id']}/turns",
        headers=headers,
        json={
            "maxun_turn_id": "99999999-9999-4999-8999-999999999997",
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    assert captured["compactor"] is not None
    assert captured["budget"] > 0


@pytest.mark.asyncio
async def test_resumable_turn_events_replay_from_sequence(app_and_db):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn_id = "99999999-9999-4999-8999-999999999996"
    body = {
        "maxun_turn_id": turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "How many rows?",
    }
    accepted = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json=body,
    )
    assert accepted.status_code == 200

    events = []
    for _ in range(20):
        replay = await _request(
            app_and_db,
            "GET",
            f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/events?after=0",
            headers=headers,
        )
        assert replay.status_code == 200
        events = replay.json()["events"]
        if replay.json()["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert [item["type"] for item in events] == [
        "turn.started",
        "query.result",
        "answer.delta",
        "turn.completed",
    ]
    assert [item["id"] for item in events] == list(range(1, len(events) + 1))

    after_query = await _request(
        app_and_db,
        "GET",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/events?after=2",
        headers=headers,
    )
    assert [item["id"] for item in after_query.json()["events"]] == [3, 4]
    assert after_query.json()["result"]["status"] == "completed"


@pytest.mark.asyncio
async def test_duplicate_turn_admission_does_not_start_duplicate_provider_execution(
    app_and_db, monkeypatch
):
    started = asyncio.Event()
    release = asyncio.Event()
    provider_calls = 0

    async def blocked_events(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        started.set()
        await release.wait()
        yield {
            "event": "SQL",
            "payload": {
                "sql": "SELECT COUNT(*) FROM data",
                "columns": ["count"],
                "rows": [{"count": 2}],
            },
        }
        yield {"event": "TEXT", "payload": {"text": "There are 2 rows."}}

    monkeypatch.setattr(adapter, "stream_graph_events", blocked_events)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn_id = "99999999-9999-4999-8999-999999999993"
    body = {
        "maxun_turn_id": turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "How many rows?",
    }

    first = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json=body,
    )
    assert first.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=1)

    duplicate = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json=body,
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["attempt"] == first.json()["attempt"] == 1
    assert provider_calls == 1

    release.set()
    for _ in range(20):
        status = await _request(
            app_and_db,
            "GET",
            f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/events?after=0",
            headers=headers,
        )
        if status.json()["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert status.json()["status"] == "completed"
    assert provider_calls == 1


@pytest.mark.asyncio
async def test_resumable_turn_rejects_maxun_turn_id_reuse(app_and_db):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn_id = "99999999-9999-4999-8999-999999999994"
    body = {
        "maxun_turn_id": turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "How many rows?",
    }
    await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json=body,
    )
    reused = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json={**body, "question": "What is the total?"},
    )
    assert reused.status_code == 409
    assert reused.json()["detail"]["code"] == "MAXUN_TURN_ID_REUSED"


@pytest.mark.asyncio
async def test_answer_deltas_are_coalesced_before_ledger_persistence(app_and_db, monkeypatch):
    async def many_text_events(**kwargs):
        yield {
            "event": "SQL",
            "payload": {
                "sql": "SELECT COUNT(*) FROM data",
                "columns": ["count"],
                "rows": [{"count": 20}],
                "truncated": False,
            },
        }
        for _ in range(20):
            yield {"event": "TEXT", "payload": {"text": "x"}}
        yield {"event": "COMPLETE", "payload": {"text": ""}}

    monkeypatch.setattr(adapter, "stream_graph_events", many_text_events)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    response = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": "99999999-9999-4999-8999-999999999994",
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    async with adapter._get_session_factory()() as session:
        from analytics_agent.db.models import MaxunTurnEvent
        from sqlalchemy import select

        event_rows = list((await session.execute(select(MaxunTurnEvent))).scalars().all())
    assert [row.event_type for row in event_rows].count("answer.delta") == 1


@pytest.mark.asyncio
async def test_resumable_turn_cancel_is_idempotent_and_terminal(app_and_db, monkeypatch):
    async def slow_events(**kwargs):
        await asyncio.sleep(0.2)
        yield {
            "event": "SQL",
            "payload": {
                "sql": "SELECT COUNT(*) FROM data",
                "columns": ["count"],
                "rows": [{"count": 2}],
            },
        }

    monkeypatch.setattr(adapter, "stream_graph_events", slow_events)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn_id = "99999999-9999-4999-8999-999999999995"
    body = {
        "maxun_turn_id": turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "How many rows?",
    }
    await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json=body,
    )
    cancel_body = {
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
    }
    cancelled = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/cancel",
        headers=headers,
        json=cancel_body,
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    repeated = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/cancel",
        headers=headers,
        json=cancel_body,
    )
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "cancelled"
    await asyncio.sleep(0.25)
    events = await _request(
        app_and_db,
        "GET",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/events?after=0",
        headers=headers,
    )
    assert events.json()["status"] == "cancelled"
    assert events.json()["events"][-1]["type"] == "turn.cancelled"


@pytest.mark.asyncio
async def test_turn_history_messages_are_correlated_once(app_and_db):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn_id = "99999999-9999-4999-8999-999999999993"
    response = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": turn_id,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    factory = adapter._get_session_factory()
    async with factory() as session:
        from analytics_agent.db.models import Message
        from sqlalchemy import select

        rows = list(
            (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [(row.event_type, row.role) for row in rows] == [
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
    ]
    assert all(row.maxun_turn_record_id for row in rows)
    assert len({row.maxun_turn_record_id for row in rows}) == 1

    replay = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
        headers=headers,
        json={
            "maxun_turn_id": turn_id,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert replay.status_code == 200
    assert replay.json()["status"] == "completed"
    async with factory() as session:
        rows_after_replay = list(
            (
                await session.execute(
                    select(Message).where(Message.conversation_id == conversation_id)
                )
            )
            .scalars()
            .all()
        )
    assert [(row.event_type, row.role) for row in rows_after_replay] == [
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
    ]


@pytest.mark.asyncio
async def test_two_turn_history_reconstructs_each_finalized_exchange_once(app_and_db, monkeypatch):
    original_build_history = adapter.build_history
    captured_histories = []

    def capture_history(*args, **kwargs):
        history = original_build_history(*args, **kwargs)
        captured_histories.append(history)
        return history

    monkeypatch.setattr(adapter, "build_history", capture_history)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]

    async def complete_turn(turn_id, question):
        response = await _request(
            app_and_db,
            "POST",
            f"/internal/maxun/conversations/{conversation_id}/turns",
            headers=headers,
            json={
                "maxun_turn_id": turn_id,
                "workspace_id": WORKSPACE_ID,
                "workspace_version": 1,
                "data_signature": SIGNATURE,
                "question": question,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    await complete_turn("99999999-9999-4999-8999-999999999991", "How many rows?")
    await complete_turn("99999999-9999-4999-8999-999999999990", "What is the next question?")

    assert [[message.type for message in history] for history in captured_histories] == [
        ["human"],
        ["human", "ai", "human"],
    ]
    assert [message.content for message in captured_histories[1]] == [
        "How many rows?",
        "There are 2 rows.",
        "What is the next question?",
    ]

    factory = adapter._get_session_factory()
    async with factory() as session:
        from analytics_agent.db.models import Message
        from sqlalchemy import select

        rows = list(
            (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [(row.event_type, row.role) for row in rows] == [
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
    ]
    assert len({row.maxun_turn_record_id for row in rows}) == 2


@pytest.mark.asyncio
async def test_recovered_turn_history_excludes_partial_attempt_events(app_and_db, monkeypatch):
    original_build_history = adapter.build_history
    captured_histories = []

    def capture_history(*args, **kwargs):
        history = original_build_history(*args, **kwargs)
        captured_histories.append(history)
        return history

    monkeypatch.setattr(adapter, "build_history", capture_history)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    first_turn_id = "99999999-9999-4999-8999-999999999989"
    first_body = {
        "maxun_turn_id": first_turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "How many rows?",
    }
    first = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json=first_body,
    )
    assert first.status_code == 200

    from analytics_agent.db.models import MaxunTurn, MaxunTurnEvent, Message
    from sqlalchemy import select

    recovered_turn_id = "99999999-9999-4999-8999-999999999988"
    recovered_body = {
        "maxun_turn_id": recovered_turn_id,
        "workspace_id": WORKSPACE_ID,
        "workspace_version": 1,
        "data_signature": SIGNATURE,
        "question": "Recover this question",
    }
    request = adapter.MaxunTurnRequest(**recovered_body)
    turn_record_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbb988"
    async with adapter._get_session_factory()() as session:
        session.add(
            MaxunTurn(
                id=turn_record_id,
                conversation_id=conversation_id,
                maxun_turn_id=recovered_turn_id,
                request_digest=adapter._request_digest(request),
                status="processing",
                attempt=1,
                next_event_sequence=3,
            )
        )
        session.add(
            Message(
                id="cccccccc-cccc-4ccc-8ccc-cccccccccc88",
                conversation_id=conversation_id,
                maxun_turn_record_id=turn_record_id,
                event_type="TEXT",
                role="user",
                payload=orjson.dumps({"text": recovered_body["question"]}).decode(),
                sequence=2,
            )
        )
        session.add(
            MaxunTurnEvent(
                id="dddddddd-dddd-4ddd-8ddd-dddddddddd88",
                turn_record_id=turn_record_id,
                sequence=1,
                event_type="turn.started",
                payload=orjson.dumps({"attempt": 1}).decode(),
            )
        )
        session.add(
            MaxunTurnEvent(
                id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeee88",
                turn_record_id=turn_record_id,
                sequence=2,
                event_type="answer.delta",
                payload=orjson.dumps(
                    {"text": "partial attempt that must not become history"}
                ).decode(),
            )
        )
        await session.commit()

    recovered = await _request(
        app_and_db,
        "PUT",
        f"/internal/maxun/conversations/{conversation_id}/turns/{recovered_turn_id}",
        headers=headers,
        json=recovered_body,
    )
    assert recovered.status_code == 200
    for _ in range(20):
        status = await _request(
            app_and_db,
            "GET",
            f"/internal/maxun/conversations/{conversation_id}/turns/{recovered_turn_id}/events?after=0",
            headers=headers,
        )
        if status.json()["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert status.json()["status"] == "completed"
    assert [message.content for message in captured_histories[-1]] == [
        "How many rows?",
        "There are 2 rows.",
        "Recover this question",
    ]
    assert all("partial attempt" not in message.content for message in captured_histories[-1])

    async with adapter._get_session_factory()() as session:
        messages = list(
            (
                await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [(message.event_type, message.role) for message in messages] == [
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
        ("TEXT", "user"),
        ("MAXUN_RESULT", "assistant"),
    ]


@pytest.mark.asyncio
async def test_sqlite_event_appends_are_serialized_by_turn_lock(app_and_db):
    from analytics_agent.db.models import Conversation, MaxunTurn
    from sqlalchemy import select

    factory = adapter._get_session_factory()
    conversation_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    turn_record_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    async with factory() as session:
        session.add(
            Conversation(
                id=conversation_id,
                title="Sequence test",
                engine_name=f"maxun:{WORKSPACE_ID}",
                created_at=adapter.datetime.now(adapter.UTC),
            )
        )
        session.add(
            MaxunTurn(
                id=turn_record_id,
                conversation_id=conversation_id,
                maxun_turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                request_digest="a" * 64,
                status="processing",
                next_event_sequence=1,
                attempt=1,
            )
        )
        await session.commit()
    await asyncio.gather(
        *(
            adapter._append_turn_event(turn_record_id, "answer.delta", {"text": str(index)})
            for index in range(20)
        )
    )
    async with factory() as session:
        from analytics_agent.db.models import MaxunTurnEvent

        events = list(
            (
                await session.execute(
                    select(MaxunTurnEvent)
                    .where(MaxunTurnEvent.turn_record_id == turn_record_id)
                    .order_by(MaxunTurnEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [event.sequence for event in events] == list(range(1, 21))


@pytest.mark.asyncio
async def test_agent_runtime_timeout_commits_terminal_state(app_and_db, monkeypatch):
    async def never_finishes(**kwargs):
        await asyncio.sleep(0.2)
        yield {
            "event": "SQL",
            "payload": {
                "sql": "SELECT COUNT(*) FROM data",
                "columns": ["count"],
                "rows": [{"count": 2}],
            },
        }

    monkeypatch.setattr(adapter, "stream_graph_events", never_finishes)
    previous_timeout = adapter.settings.maxun_turn_runtime_seconds
    adapter.settings.maxun_turn_runtime_seconds = 0.05
    try:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        created = await _request(
            app_and_db,
            "POST",
            "/internal/maxun/conversations",
            headers=headers,
            json={
                "workspace_id": WORKSPACE_ID,
                "workspace_version": 1,
                "data_signature": SIGNATURE,
            },
        )
        conversation_id = created.json()["conversation_id"]
        turn_id = "99999999-9999-4999-8999-999999999992"
        body = {
            "maxun_turn_id": turn_id,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        }
        accepted = await _request(
            app_and_db,
            "PUT",
            f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}",
            headers=headers,
            json=body,
        )
        assert accepted.status_code == 200
        for _ in range(30):
            status = await _request(
                app_and_db,
                "GET",
                f"/internal/maxun/conversations/{conversation_id}/turns/{turn_id}/events?after=0",
                headers=headers,
            )
            if status.json()["status"] != "processing":
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "error"
        assert status.json()["result"]["error"]["code"] == "MAXUN_TURN_TIMEOUT"
        assert status.json()["events"][-1]["type"] == "turn.failed"
    finally:
        adapter.settings.maxun_turn_runtime_seconds = previous_timeout


@pytest.mark.asyncio
async def test_internal_maxun_conversation_delete_is_idempotent(app_and_db):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    turn = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": TURN_ID,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert turn.status_code == 200

    deleted = await _request(
        app_and_db,
        "DELETE",
        f"/internal/maxun/conversations/{conversation_id}",
        headers=headers,
    )
    assert deleted.status_code == 204
    repeated = await _request(
        app_and_db,
        "DELETE",
        f"/internal/maxun/conversations/{conversation_id}",
        headers=headers,
    )
    assert repeated.status_code == 204
    factory = adapter._get_session_factory()
    async with factory() as session:
        from analytics_agent.db.models import MaxunTurn, MaxunTurnEvent, Message
        from analytics_agent.db.repository import ConversationRepo
        from sqlalchemy import select

        assert await ConversationRepo(session).get(conversation_id) is None
        assert (
            await session.execute(
                select(MaxunTurn).where(MaxunTurn.conversation_id == conversation_id)
            )
        ).scalars().all() == []
        assert (
            await session.execute(select(Message).where(Message.conversation_id == conversation_id))
        ).scalars().all() == []
        assert (await session.execute(select(MaxunTurnEvent))).scalars().all() == []


@pytest.mark.asyncio
async def test_lock_registry_preserves_queued_turn_waiters():
    key = "lock-waiter-test"
    lock = adapter._lock_for(key)
    await lock.acquire()
    waiter = asyncio.create_task(lock.acquire())
    await asyncio.sleep(0)
    adapter._maybe_drop_lock(adapter._turn_locks, key, lock)
    assert adapter._turn_locks.get(key) is lock
    lock.release()
    await waiter
    lock.release()
    adapter._maybe_drop_lock(adapter._turn_locks, key, lock)
    assert key not in adapter._turn_locks


def test_phase5_result_history_reconstructs_finalized_exchange():
    history = adapter.build_history(
        [
            SimpleNamespace(
                role="user",
                event_type="TEXT",
                payload=orjson.dumps({"text": "How many rows?"}).decode(),
                id="user-1",
            ),
            SimpleNamespace(
                role="assistant",
                event_type="MAXUN_RESULT",
                payload=orjson.dumps(
                    {
                        "status": "completed",
                        "answer": "There are 120 rows.",
                        "sql": "SELECT COUNT(*) FROM data",
                        "columns": ["count"],
                        "rows": [{"count": 120}],
                        "truncated": False,
                    }
                ).decode(),
                id="result-1",
            ),
        ],
        "What percentage are active?",
    )
    assert [message.type for message in history] == ["human", "ai", "human"]
    assert history[0].content == "How many rows?"
    assert history[1].content == "There are 120 rows."
    assert history[2].content == "What percentage are active?"


def test_event_payload_accepts_exact_boundary_and_rejects_one_byte_over():
    prefix = len(orjson.dumps({"text": ""}))
    exact = {"text": "x" * (adapter._MAX_EVENT_PAYLOAD_BYTES - prefix)}
    assert len(orjson.dumps(exact)) == adapter._MAX_EVENT_PAYLOAD_BYTES
    assert len(adapter._encoded_event_payload(exact).encode()) == adapter._MAX_EVENT_PAYLOAD_BYTES
    with pytest.raises(adapter.MaxunQueryError) as error:
        adapter._encoded_event_payload({"text": exact["text"] + "x"})
    assert error.value.code == "MAXUN_RESULT_LIMIT"


def test_public_result_preserves_one_envelope_for_query_and_terminal_result():
    bounded = adapter._bounded_public_result(
        {
            "sql": "SELECT * FROM data",
            "columns": ["value"],
            "rows": [{"value": "x" * 500} for _ in range(20)],
            "truncated": False,
        },
        answer_budget="x" * 12_000,
    )
    terminal = {
        "status": "completed",
        "answer": "x" * 12_000,
        **bounded,
        "error": None,
    }
    assert len(orjson.dumps(bounded)) <= adapter._MAX_EVENT_PAYLOAD_BYTES
    assert len(orjson.dumps(terminal)) <= adapter._MAX_EVENT_PAYLOAD_BYTES
    assert bounded["truncated"] is False


def test_public_result_is_byte_bounded_with_deterministic_truncation():
    bounded = adapter._bounded_public_result(
        {
            "sql": "SELECT * FROM data",
            "columns": ["value"],
            "rows": [{"value": "x" * 2_000} for _ in range(500)],
            "truncated": False,
        },
        answer_budget="x" * 12_000,
    )
    assert bounded["truncated"] is True
    assert len(bounded["rows"]) < 500
    assert len(orjson.dumps(bounded)) <= adapter._MAX_EVENT_PAYLOAD_BYTES


def test_result_requires_successful_sql():
    assert adapter._result_from_events(
        [
            {"event": "TEXT", "payload": {"text": "42"}},
            {"event": "COMPLETE", "payload": {"text": "42"}},
        ]
    ) == {
        "status": "error",
        "answer": "",
        "sql": None,
        "columns": [],
        "rows": [],
        "truncated": False,
        "error": {
            "code": "MAXUN_QUERY_REQUIRED",
            "message": "The workspace question could not be completed",
        },
    }


def test_result_deduplicates_repeated_sql_events_but_rejects_distinct_queries():
    duplicate = adapter._result_from_events(
        [
            {
                "event": "SQL",
                "payload": {
                    "sql": "SELECT COUNT(*) FROM data",
                    "columns": ["count"],
                    "rows": [{"count": 2}],
                },
            },
            {
                "event": "SQL",
                "payload": {
                    "sql": "SELECT COUNT(*) FROM data",
                    "columns": ["count"],
                    "rows": [{"count": 2}],
                },
            },
        ]
    )
    assert duplicate["status"] == "completed"
    distinct = adapter._result_from_events(
        [
            {
                "event": "SQL",
                "payload": {
                    "sql": "SELECT COUNT(*) FROM data",
                    "columns": ["count"],
                    "rows": [{"count": 2}],
                },
            },
            {
                "event": "SQL",
                "payload": {
                    "sql": "SELECT SUM(value) FROM data",
                    "columns": ["sum"],
                    "rows": [{"sum": 3}],
                },
            },
        ]
    )
    assert distinct["status"] == "error"
    assert distinct["error"]["code"] == "MAXUN_QUERY_LIMIT"


def test_result_allows_recovery_after_failed_sql():
    result = adapter._result_from_events(
        [
            {"event": "TOOL_RESULT", "payload": {"tool_name": "execute_sql", "is_error": True}},
            {
                "event": "SQL",
                "payload": {
                    "sql": "SELECT COUNT(*) FROM data",
                    "columns": ["count"],
                    "rows": [{"count": 2}],
                },
            },
            {"event": "TEXT", "payload": {"text": "There are 2 rows."}},
            {"event": "COMPLETE", "payload": {"text": "There are 2 rows."}},
        ]
    )
    assert result["status"] == "completed"
    assert result["sql"] == "SELECT COUNT(*) FROM data"


@pytest.mark.asyncio
async def test_text_only_provider_completion_is_returned_as_error(app_and_db, monkeypatch):
    async def text_only_events(**kwargs):
        yield {"event": "TEXT", "payload": {"text": "The answer is 42."}}
        yield {"event": "COMPLETE", "payload": {"text": "The answer is 42."}}

    monkeypatch.setattr(adapter, "stream_graph_events", text_only_events)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    created = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE},
    )
    conversation_id = created.json()["conversation_id"]
    response = await _request(
        app_and_db,
        "POST",
        f"/internal/maxun/conversations/{conversation_id}/turns",
        headers=headers,
        json={
            "maxun_turn_id": "99999999-9999-4999-8999-999999999998",
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "MAXUN_QUERY_REQUIRED"


@pytest.mark.asyncio
async def test_internal_route_rejects_malformed_snapshot_and_unknown_conversation(app_and_db):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations",
        headers=headers,
        json={"workspace_id": "../private", "workspace_version": 1, "data_signature": SIGNATURE},
    )
    assert response.status_code == 400
    assert "private" not in response.text

    response = await _request(
        app_and_db,
        "POST",
        "/internal/maxun/conversations/not-a-uuid/turns",
        headers=headers,
        json={
            "maxun_turn_id": TURN_ID,
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "x",
        },
    )
    assert response.status_code == 404
