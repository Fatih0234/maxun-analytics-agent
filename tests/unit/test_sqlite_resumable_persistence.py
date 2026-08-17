from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from analytics_agent.db.models import Base, Conversation, MaxunTurn, MaxunTurnEvent
from analytics_agent.db.repository import MaxunTurnRepo
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.mark.asyncio
async def test_file_backed_sqlite_turn_ledger_survives_engine_restart(tmp_path: Path):
    db_path = tmp_path / "resumable.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    conversation_id = str(uuid4())
    turn_id = str(uuid4())

    engine = create_async_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(
            Conversation(id=conversation_id, title="Restart", engine_name="maxun:workspace")
        )
        session.add(
            MaxunTurn(
                id=turn_id,
                conversation_id=conversation_id,
                maxun_turn_id=str(uuid4()),
                request_digest="a" * 64,
                status="processing",
                attempt=2,
                next_event_sequence=1,
            )
        )
        await session.commit()
    async with sessions() as session:
        await MaxunTurnRepo(session).append_event(turn_id, "turn.started", '{"attempt":2}')
        await MaxunTurnRepo(session).append_event(turn_id, "query.result", '{"rows":[]}')
    await engine.dispose()

    restarted = create_async_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
    restarted_sessions = async_sessionmaker(restarted, expire_on_commit=False)
    async with restarted_sessions() as session:
        turn = await session.get(MaxunTurn, turn_id)
        events = list(
            (
                await session.execute(
                    select(MaxunTurnEvent)
                    .where(MaxunTurnEvent.turn_record_id == turn_id)
                    .order_by(MaxunTurnEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert turn is not None
    assert turn.attempt == 2
    assert turn.next_event_sequence == 3
    assert [event.sequence for event in events] == [1, 2]
    await restarted.dispose()


@pytest.mark.asyncio
async def test_file_backed_sqlite_turn_event_limit_reserves_terminal_slot(tmp_path: Path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'limit.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    conversation_id = str(uuid4())
    turn_id = str(uuid4())
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(Conversation(id=conversation_id, title="Limit", engine_name="maxun:workspace"))
        session.add(
            MaxunTurn(
                id=turn_id,
                conversation_id=conversation_id,
                maxun_turn_id=str(uuid4()),
                request_digest="b" * 64,
                status="processing",
                next_event_sequence=512,
            )
        )
        await session.commit()
    async with sessions() as session:
        with pytest.raises(ValueError, match="event limit"):
            await MaxunTurnRepo(session).append_event(turn_id, "answer.delta", '{"text":"x"}')
    async with sessions() as session:
        event = await MaxunTurnRepo(session).append_event(
            turn_id, "turn.failed", '{"status":"error"}'
        )
        assert event.sequence == 512
    await engine.dispose()
