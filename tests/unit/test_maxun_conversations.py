from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
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
            "payload": {"tool_name": "execute_sql", "tool_input": {"sql": "SELECT COUNT(*) FROM data"}},
        }
        yield {
            "event": "SQL",
            "payload": {"sql": "SELECT COUNT(*) FROM data", "columns": ["count"], "rows": [{"count": 2}], "truncated": False},
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
        from analytics_agent.db.repository import ConversationRepo

        assert await ConversationRepo(session).get(conversation_id) is None


def test_result_requires_successful_sql():
    assert adapter._result_from_events([
        {"event": "TEXT", "payload": {"text": "42"}},
        {"event": "COMPLETE", "payload": {"text": "42"}},
    ]) == {
        "status": "error",
        "answer": "42",
        "sql": None,
        "columns": [],
        "rows": [],
        "truncated": False,
        "error": {"code": "MAXUN_QUERY_REQUIRED", "message": "The workspace question could not be completed"},
    }


def test_result_deduplicates_repeated_sql_events_but_rejects_distinct_queries():
    duplicate = adapter._result_from_events([
        {"event": "SQL", "payload": {"sql": "SELECT COUNT(*) FROM data", "columns": ["count"], "rows": [{"count": 2}]}},
        {"event": "SQL", "payload": {"sql": "SELECT COUNT(*) FROM data", "columns": ["count"], "rows": [{"count": 2}]}},
    ])
    assert duplicate["status"] == "completed"
    distinct = adapter._result_from_events([
        {"event": "SQL", "payload": {"sql": "SELECT COUNT(*) FROM data", "columns": ["count"], "rows": [{"count": 2}]}},
        {"event": "SQL", "payload": {"sql": "SELECT SUM(value) FROM data", "columns": ["sum"], "rows": [{"sum": 3}]}},
    ])
    assert distinct["status"] == "error"
    assert distinct["error"]["code"] == "MAXUN_QUERY_LIMIT"


def test_result_allows_recovery_after_failed_sql():
    result = adapter._result_from_events([
        {"event": "TOOL_RESULT", "payload": {"tool_name": "execute_sql", "is_error": True}},
        {"event": "SQL", "payload": {"sql": "SELECT COUNT(*) FROM data", "columns": ["count"], "rows": [{"count": 2}]}},
        {"event": "TEXT", "payload": {"text": "There are 2 rows."}},
        {"event": "COMPLETE", "payload": {"text": "There are 2 rows."}},
    ])
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
        json={"maxun_turn_id": TURN_ID, "workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE, "question": "x"},
    )
    assert response.status_code == 404
