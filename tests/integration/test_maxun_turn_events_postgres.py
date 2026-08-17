from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from analytics_agent import bootstrap
from analytics_agent.config import settings
from analytics_agent.db import base
from analytics_agent.db.models import Conversation, MaxunTurn
from analytics_agent.db.repository import MaxunTurnRepo


@pytest.fixture(scope="module")
def postgres_database():
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql+"):
        pytest.skip("DATABASE_URL must point to PostgreSQL for this integration test")
    original_url = settings.database_url
    original_engine = base._engine
    original_factory = base._AsyncSessionFactory
    settings.database_url = url
    base._engine = None
    base._AsyncSessionFactory = None
    repo_root = Path(__file__).resolve().parents[2]
    old_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        bootstrap.run_migrations()
        yield url
    finally:
        os.chdir(old_cwd)
        if base._engine is not None:
            asyncio.run(base._engine.dispose())
        settings.database_url = original_url
        base._engine = original_engine
        base._AsyncSessionFactory = original_factory


@pytest.mark.asyncio
async def test_postgres_turn_event_sequence_is_serialized(postgres_database):
    factory = base._get_session_factory()
    conversation_id = str(uuid4())
    turn_id = str(uuid4())
    async with factory() as session:
        session.add(
            Conversation(
                id=conversation_id,
                title="Phase 5 sequence test",
                engine_name="maxun:11111111-1111-4111-8111-111111111111",
            )
        )
        session.add(
            MaxunTurn(
                id=turn_id,
                conversation_id=conversation_id,
                maxun_turn_id=str(uuid4()),
                request_digest="a" * 64,
                status="processing",
                result_json=None,
                attempt=1,
                next_event_sequence=1,
            )
        )
        await session.commit()

    async def append(index: int) -> int:
        async with factory() as session:
            event = await MaxunTurnRepo(session).append_event(
                turn_id,
                "answer.delta",
                json.dumps({"text": str(index)}),
            )
            return event.sequence

    sequences = await asyncio.gather(*(append(index) for index in range(20)))
    assert sorted(sequences) == list(range(1, 21))
    async with factory() as session:
        events = await MaxunTurnRepo(session).list_events(turn_id, after_sequence=0, limit=100)
        assert [event.sequence for event in events] == list(range(1, 21))
