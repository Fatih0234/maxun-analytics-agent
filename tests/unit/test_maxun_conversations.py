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
            "workspace_id": WORKSPACE_ID,
            "workspace_version": 1,
            "data_signature": SIGNATURE,
            "question": "How many rows?",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "answer": "There are 2 rows.",
        "sql": "SELECT COUNT(*) FROM data",
        "columns": ["count"],
        "rows": [{"count": 2}],
        "truncated": False,
    }


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
        json={"workspace_id": WORKSPACE_ID, "workspace_version": 1, "data_signature": SIGNATURE, "question": "x"},
    )
    assert response.status_code == 404
