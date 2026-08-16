"""Private Maxun-owned conversation adapter.

The Agent stores conversation history, but Maxun remains the owner of the
conversation and the authority for workspace authorization and provenance.
These routes are not browser APIs and accept only the narrow internal bearer
credential shared by the Maxun backend.
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from analytics_agent.agent.graph import build_graph
from analytics_agent.agent.history import build_history
from analytics_agent.agent.streaming import stream_graph_events
from analytics_agent.config import settings
from analytics_agent.db.base import _get_session_factory
from analytics_agent.db.models import Conversation, Message
from analytics_agent.db.repository import ConversationRepo, MessageRepo
from analytics_agent.engines.maxun.engine import MaxunQueryError
from analytics_agent.engines.resolver import resolve_engine
from analytics_agent.maxun.materialization import configured_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal/maxun/conversations", tags=["maxun-conversations"])

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ANSWER_CHARS = 12_000
_MAX_QUESTION_CHARS = 4_000

# The supported deployment is single-replica for this phase. The lock prevents
# duplicate in-flight turns inside one Agent process; Maxun's idempotency and
# database state remain the public authority.
_turn_locks: dict[str, Any] = {}


class MaxunConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    workspace_version: int = Field(ge=1)
    data_signature: str
    title: str = Field(default="Workspace analysis", max_length=255)


class MaxunTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    workspace_version: int = Field(ge=1)
    data_signature: str
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


def _internal_authorized(authorization: str | None) -> None:
    expected = configured_token()
    if (
        not expected
        or not authorization
        or not authorization.startswith("Bearer ")
        or not hmac.compare_digest(authorization[7:].strip(), expected)
    ):
        raise HTTPException(status_code=503, detail={"code": "MAXUN_INTERNAL_UNAVAILABLE"})


def _validate_snapshot(workspace_id: str, version: int, signature: str) -> None:
    if not _UUID_RE.fullmatch(workspace_id) or workspace_id != workspace_id.lower():
        raise HTTPException(status_code=400, detail={"code": "MAXUN_SNAPSHOT_INVALID"})
    if version < 1 or not _SIGNATURE_RE.fullmatch(signature):
        raise HTTPException(status_code=400, detail={"code": "MAXUN_SNAPSHOT_INVALID"})


def _safe_error(error: BaseException) -> dict[str, str]:
    if isinstance(error, MaxunQueryError):
        return {"code": error.code, "message": error.message}
    return {"code": "MAXUN_TURN_FAILED", "message": "The workspace question could not be completed"}


def _new_message(
    conversation_id: str,
    event_type: str,
    role: str,
    payload: dict[str, Any],
    sequence: int,
) -> Message:
    return Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        event_type=event_type,
        role=role,
        payload=orjson.dumps(payload).decode(),
        sequence=sequence,
        created_at=datetime.now(UTC),
    )


def _lock_for(conversation_id: str):
    import asyncio

    lock = _turn_locks.get(conversation_id)
    if lock is None:
        lock = asyncio.Lock()
        _turn_locks[conversation_id] = lock
    return lock


def _result_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    answer_parts: list[str] = []
    sql_result: dict[str, Any] | None = None
    failed = False
    for event in events:
        event_type = event.get("event")
        payload = event.get("payload") or {}
        if event_type == "TEXT":
            text = payload.get("text")
            if isinstance(text, str):
                answer_parts.append(text)
        elif event_type == "SQL":
            sql_result = {
                "sql": payload.get("sql", ""),
                "columns": payload.get("columns", []),
                "rows": payload.get("rows", []),
                "truncated": bool(payload.get("truncated", False)),
            }
        elif event_type == "ERROR":
            failed = True

    answer = "".join(answer_parts).strip()[:_MAX_ANSWER_CHARS]
    return {
        "status": "error" if failed else "completed",
        "answer": answer,
        "sql": sql_result.get("sql") if sql_result else None,
        "columns": sql_result.get("columns", []) if sql_result else [],
        "rows": sql_result.get("rows", []) if sql_result else [],
        "truncated": sql_result.get("truncated", False) if sql_result else False,
    }


async def _run_turn(
    conversation_id: str,
    request: MaxunTurnRequest,
) -> dict[str, Any]:
    factory = _get_session_factory()
    async with factory() as session:
        repo = ConversationRepo(session)
        conversation = await repo.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})
        expected_engine = f"maxun:{request.workspace_id}"
        if conversation.engine_name != expected_engine:
            raise HTTPException(status_code=409, detail={"code": "MAXUN_CONVERSATION_BINDING_MISMATCH"})

        message_repo = MessageRepo(session)
        prior_messages = await message_repo.list_for_conversation(conversation_id)
        sequence = len(prior_messages)
        session.add(_new_message(
            conversation_id,
            "TEXT",
            "user",
            {"text": request.question.strip()},
            sequence,
        ))
        sequence += 1

        engine = None
        events: list[dict[str, Any]] = []
        try:
            engine = await resolve_engine(
                expected_engine,
                session,
                maxun_workspace_signature=request.data_signature,
                maxun_workspace_version=request.workspace_version,
            )
            history = build_history(
                prior_messages,
                request.question.strip(),
                max_history_tokens=settings.max_history_tokens,
            )
            graph = build_graph(
                engine_name=expected_engine,
                engine=engine,
                engine_tools=engine.get_tools(),
                context_tools=[],
                disabled_tools={"create_chart"},
                enabled_mutations=set(),
                maxun_readonly=True,
            )
            async for event in stream_graph_events(
                graph=graph,
                user_text=request.question.strip(),
                conversation_id=conversation_id,
                engine_name=expected_engine,
                keepalive_interval=settings.sse_keepalive_interval,
                history=history,
            ):
                event_type = event.get("event")
                if event_type in {"KEEPALIVE", "USAGE", "CHART"}:
                    continue
                if event_type == "ERROR":
                    # Do not persist or return provider/path/parser internals.
                    event = {
                        "event": "ERROR",
                        "conversation_id": conversation_id,
                        "message_id": str(uuid.uuid4()),
                        "payload": {"error": "The workspace question could not be completed"},
                    }
                events.append(event)
                if event_type in {"TEXT", "TOOL_CALL", "TOOL_RESULT", "SQL", "ERROR", "COMPLETE"}:
                    role = "assistant"
                    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                    session.add(_new_message(conversation_id, event_type, role, payload, sequence))
                    sequence += 1
        except Exception as error:
            safe = _safe_error(error)
            logger.info("Maxun internal turn failed: %s", safe["code"])
            failed_event = {
                "event": "ERROR",
                "conversation_id": conversation_id,
                "message_id": str(uuid.uuid4()),
                "payload": {"error": "The workspace question could not be completed"},
            }
            events.append(failed_event)
            session.add(_new_message(
                conversation_id,
                "ERROR",
                "assistant",
                {"error": "The workspace question could not be completed"},
                sequence,
            ))
        finally:
            if engine is not None:
                with contextlib.suppress(Exception):
                    await engine.aclose()

        result = _result_from_events(events)
        if result["status"] == "error":
            result["error"] = {"code": "MAXUN_TURN_FAILED", "message": "The workspace question could not be completed"}
        session.add(
            # Touching the row here avoids a second commit through the generic
            # repository and keeps the turn's history atomic.
            _new_message(
                conversation_id,
                "MAXUN_RESULT",
                "assistant",
                {
                    "status": result["status"],
                    "answer": result["answer"],
                    "sql": result["sql"],
                    "columns": result["columns"],
                    "rows": result["rows"],
                    "truncated": result["truncated"],
                },
                sequence,
            )
        )
        await session.commit()
        return result


@router.post("", status_code=201)
async def create_conversation(
    body: MaxunConversationCreate,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    _validate_snapshot(body.workspace_id, body.workspace_version, body.data_signature)

    factory = _get_session_factory()
    async with factory() as session:
        engine = None
        try:
            engine = await resolve_engine(
                f"maxun:{body.workspace_id}",
                session,
                maxun_workspace_signature=body.data_signature,
                maxun_workspace_version=body.workspace_version,
            )
            conversation = Conversation(
                id=str(uuid.uuid4()),
                title=body.title.strip() or "Workspace analysis",
                engine_name=f"maxun:{body.workspace_id}",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            session.add(conversation)
            await session.commit()
            return {
                "conversation_id": conversation.id,
                "status": "ready",
            }
        except HTTPException:
            raise
        except Exception as error:
            logger.info("Maxun internal conversation creation failed: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail={"code": "MAXUN_WORKSPACE_UNAVAILABLE"}) from error
        finally:
            if engine is not None:
                with contextlib.suppress(Exception):
                    await engine.aclose()


@router.post("/{conversation_id}/turns")
async def create_turn(
    conversation_id: str,
    body: MaxunTurnRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    _validate_snapshot(body.workspace_id, body.workspace_version, body.data_signature)
    if not _UUID_RE.fullmatch(conversation_id):
        raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})

    lock = _lock_for(conversation_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail={"code": "MAXUN_TURN_IN_PROGRESS"})
    try:
        async with lock:
            return await _run_turn(conversation_id, body)
    finally:
        _turn_locks.pop(conversation_id, None)


__all__ = ["router"]
