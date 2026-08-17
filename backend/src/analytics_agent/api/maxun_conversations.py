"""Private Maxun-owned conversation adapter.

The Agent stores conversation history, but Maxun remains the owner of the
conversation and the authority for workspace authorization and provenance.
These routes are not browser APIs and accept only the narrow internal bearer
credential shared by the Maxun backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from analytics_agent.agent.graph import build_graph
from analytics_agent.agent.history import build_history
from analytics_agent.agent.streaming import stream_graph_events
from analytics_agent.config import settings
from analytics_agent.db.base import _get_session_factory
from analytics_agent.db.models import Conversation, MaxunTurn, MaxunTurnEvent, Message
from analytics_agent.db.repository import ConversationRepo, MaxunTurnRepo, MessageRepo
from analytics_agent.engines.maxun.engine import MaxunQueryError
from analytics_agent.engines.resolver import resolve_engine
from analytics_agent.maxun.materialization import configured_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal/maxun/conversations", tags=["maxun-conversations"])

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ANSWER_CHARS = 12_000
_MAX_QUESTION_CHARS = 4_000
_MAXUN_TURN_PROCESSING_TTL_SECONDS = 15 * 60
_MAX_EVENT_PAYLOAD_BYTES = 256 * 1024


def _turn_concurrency_from_env() -> int:
    try:
        value = int(os.environ.get("MAXUN_TURN_CONCURRENCY", "2"))
    except ValueError as error:
        raise ValueError("MAXUN_TURN_CONCURRENCY must be an integer") from error
    if not 1 <= value <= 32:
        raise ValueError("MAXUN_TURN_CONCURRENCY must be between 1 and 32")
    return value


_MAX_TURN_CONCURRENCY = _turn_concurrency_from_env()
_PUBLIC_EVENT_TYPES = {
    "turn.started",
    "turn.reset",
    "query.result",
    "answer.delta",
    "turn.completed",
    "turn.failed",
    "turn.cancelled",
}

# The supported deployment is single-replica for this phase. The lock prevents
# duplicate in-flight turns inside one Agent process; Maxun's idempotency and
# database state remain the public authority.
_turn_locks: dict[str, Any] = {}
_turn_tasks: dict[str, asyncio.Task[Any]] = {}
_turn_capacity = asyncio.Semaphore(_MAX_TURN_CONCURRENCY)
_active_engines: dict[str, Any] = {}


class MaxunConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    workspace_version: int = Field(ge=1)
    data_signature: str
    title: str = Field(default="Workspace analysis", max_length=255)


class MaxunTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maxun_turn_id: str = Field(min_length=1, max_length=255)
    workspace_id: str
    workspace_version: int = Field(ge=1)
    data_signature: str
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class MaxunTurnCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    workspace_version: int = Field(ge=1)
    data_signature: str


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


def _request_digest(request: MaxunTurnRequest) -> str:
    canonical = "\x1f".join(
        (
            request.workspace_id,
            str(request.workspace_version),
            request.data_signature,
            request.question.strip(),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stored_turn_result(record: MaxunTurn) -> dict[str, Any] | None:
    if record.status not in {"completed", "error", "cancelled"} or not record.result_json:
        return None
    try:
        value = orjson.loads(record.result_json)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


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


async def _mock_maxun_events(
    engine: Any,
    conversation_id: str,
    question: str,
) -> AsyncIterator[dict[str, Any]]:
    """Deterministic test-only path used by local smoke/e2e environments.

    It still invokes the real workspace tool and therefore exercises the
    materialization, AST policy, bounded DuckDB result, adapter, and Maxun
    transport without requiring a provider key. Production deployments must
    leave MOCK_LLM unset.
    """

    lowered = question.casefold()
    sql = (
        'SELECT SUM("Price") AS total FROM data'
        if "sum" in lowered or "total" in lowered
        else "SELECT COUNT(*) AS count FROM data"
    )
    execute = next((item for item in engine.get_tools() if item.name == "execute_sql"), None)
    if execute is None:
        raise MaxunQueryError("MAXUN_QUERY_FAILED", "The workspace query could not be completed")
    raw = await asyncio.to_thread(execute.invoke, {"sql": sql})
    result = orjson.loads(raw)
    if result.get("error"):
        yield {
            "event": "ERROR",
            "conversation_id": conversation_id,
            "payload": {"error": "The workspace question could not be completed"},
        }
        return
    rows = result.get("rows", [])
    value = (
        rows[0].get("total" if "sum" in lowered or "total" in lowered else "count")
        if rows
        else None
    )
    answer = f"The result is {value}."
    yield {
        "event": "SQL",
        "conversation_id": conversation_id,
        "payload": {
            "sql": sql,
            "columns": result.get("columns", []),
            "rows": rows,
            "truncated": result.get("truncated", False),
        },
    }
    yield {"event": "TEXT", "conversation_id": conversation_id, "payload": {"text": answer}}
    yield {"event": "COMPLETE", "conversation_id": conversation_id, "payload": {"text": answer}}


def _configure_maxun_turn_budget(engine: Any) -> None:
    configure = getattr(engine, "configure_turn_budget", None)
    if callable(configure):
        configure(max_query_tools=3, max_sql_executions=1)


def _encoded_event_payload(payload: dict[str, Any]) -> str:
    try:
        encoded = orjson.dumps(payload)
    except Exception:
        encoded = b'{"error":"The workspace question could not be completed"}'
    if len(encoded) > _MAX_EVENT_PAYLOAD_BYTES:
        encoded = b'{"error":"The workspace event was too large"}'
    return encoded.decode()


def _append_event_in_session(
    session: Any,
    turn: MaxunTurn,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if event_type not in _PUBLIC_EVENT_TYPES:
        return
    sequence = max(1, int(turn.next_event_sequence or 1))
    turn.next_event_sequence = sequence + 1
    turn.updated_at = datetime.now(UTC)
    session.add(
        MaxunTurnEvent(
            id=str(uuid.uuid4()),
            turn_record_id=turn.id,
            sequence=sequence,
            event_type=event_type,
            payload=_encoded_event_payload(payload),
            created_at=datetime.now(UTC),
        )
    )


async def _append_turn_event(
    turn_record_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if event_type not in _PUBLIC_EVENT_TYPES:
        return
    factory = _get_session_factory()
    async with factory() as session:
        await MaxunTurnRepo(session).append_event(
            turn_record_id,
            event_type,
            _encoded_event_payload(payload),
        )


async def _turn_is_cancel_requested(turn_record_id: str) -> bool:
    factory = _get_session_factory()
    async with factory() as session:
        turn = await MaxunTurnRepo(session).get_by_id(turn_record_id)
        return bool(turn and turn.cancel_requested_at is not None)


async def _mark_turn_started(turn_record_id: str, attempt: int, recovered: bool) -> None:
    factory = _get_session_factory()
    async with factory() as session:
        turn = await MaxunTurnRepo(session).get_by_id(turn_record_id)
        if turn is None or turn.status != "processing":
            return
        turn.started_at = datetime.now(UTC)
        turn.updated_at = datetime.now(UTC)
        await session.commit()
    if recovered:
        await _append_turn_event(
            turn_record_id,
            "turn.reset",
            {"attempt": attempt, "reason": "execution_recovered"},
        )
    await _append_turn_event(
        turn_record_id,
        "turn.started",
        {"attempt": attempt},
    )


def _cancelled_result() -> dict[str, Any]:
    return {
        "status": "cancelled",
        "answer": "",
        "sql": None,
        "columns": [],
        "rows": [],
        "truncated": False,
        "error": {
            "code": "MAXUN_TURN_CANCELLED",
            "message": "The workspace question was cancelled",
        },
    }


def _result_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    answer_parts: list[str] = []
    sql_result: dict[str, Any] | None = None
    sql_fingerprints: set[str] = set()
    successful_sql_count = 0
    tool_error = False
    failed = False
    failure_code = "MAXUN_TURN_FAILED"
    for event in events:
        event_type = event.get("event")
        payload = event.get("payload") or {}
        if event_type == "TEXT":
            text = payload.get("text")
            if isinstance(text, str) and successful_sql_count > 0:
                answer_parts.append(text)
        elif event_type == "SQL":
            sql = payload.get("sql")
            if isinstance(sql, str) and sql.strip():
                candidate = {
                    "sql": sql,
                    "columns": payload.get("columns", []),
                    "rows": payload.get("rows", []),
                    "truncated": bool(payload.get("truncated", False)),
                }
                fingerprint = repr(candidate)
                if fingerprint not in sql_fingerprints:
                    sql_fingerprints.add(fingerprint)
                    successful_sql_count += 1
                    sql_result = candidate
        elif event_type == "TOOL_RESULT" and payload.get("is_error"):
            tool_error = True
            failure_code = "MAXUN_QUERY_FAILED"
        elif event_type == "ERROR":
            failed = True
            failure_code = "MAXUN_TURN_FAILED"

    if not failed and successful_sql_count == 0:
        failed = True
        failure_code = "MAXUN_QUERY_FAILED" if tool_error else "MAXUN_QUERY_REQUIRED"
    elif not failed and successful_sql_count > 1:
        failed = True
        failure_code = "MAXUN_QUERY_LIMIT"

    answer = "".join(answer_parts).strip()[:_MAX_ANSWER_CHARS]
    result = {
        "status": "error" if failed else "completed",
        "answer": answer,
        "sql": sql_result.get("sql") if sql_result else None,
        "columns": sql_result.get("columns", []) if sql_result else [],
        "rows": sql_result.get("rows", []) if sql_result else [],
        "truncated": sql_result.get("truncated", False) if sql_result else False,
    }
    if failed:
        result["error"] = {
            "code": failure_code,
            "message": "The workspace question could not be completed",
        }
    return result


async def _ensure_turn_record(
    conversation_id: str,
    request: MaxunTurnRequest,
    *,
    recover_processing: bool | None = True,
) -> tuple[MaxunTurn, dict[str, Any] | None, bool]:
    """Create or recover the durable Agent turn record.

    ``recover_processing`` is used by the background ensure route when the
    process has no local task for a processing record. The stable Maxun turn ID
    remains the idempotency key; recovery increments the execution attempt
    instead of creating another customer turn.
    """
    factory = _get_session_factory()
    async with factory() as session:
        repo = ConversationRepo(session)
        conversation = await repo.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})
        expected_engine = f"maxun:{request.workspace_id}"
        if conversation.engine_name != expected_engine:
            raise HTTPException(
                status_code=409, detail={"code": "MAXUN_CONVERSATION_BINDING_MISMATCH"}
            )

        maxun_turn_repo = MaxunTurnRepo(session)
        request_digest = _request_digest(request)
        existing_turn = await maxun_turn_repo.get(conversation_id, request.maxun_turn_id)
        if existing_turn:
            if existing_turn.request_digest != request_digest:
                raise HTTPException(status_code=409, detail={"code": "MAXUN_TURN_ID_REUSED"})
            replay = _stored_turn_result(existing_turn)
            if replay is not None:
                return existing_turn, replay, False
            if recover_processing is None:
                return existing_turn, None, False
            if not recover_processing:
                updated_at = existing_turn.updated_at
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=UTC)
                age = (datetime.now(UTC) - updated_at).total_seconds()
                if age < _MAXUN_TURN_PROCESSING_TTL_SECONDS:
                    raise HTTPException(status_code=409, detail={"code": "MAXUN_TURN_IN_PROGRESS"})
            existing_turn.status = "processing"
            existing_turn.result_json = None
            existing_turn.attempt = max(1, existing_turn.attempt) + 1
            existing_turn.cancel_requested_at = None
            existing_turn.started_at = None
            existing_turn.finished_at = None
            existing_turn.updated_at = datetime.now(UTC)
            await session.commit()
            return existing_turn, None, True

        maxun_turn = MaxunTurn(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            maxun_turn_id=request.maxun_turn_id,
            request_digest=request_digest,
            status="processing",
            result_json=None,
            attempt=1,
            cancel_requested_at=None,
            started_at=None,
            finished_at=None,
            next_event_sequence=1,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        try:
            async with session.begin_nested():
                session.add(maxun_turn)
                await session.flush()
        except IntegrityError:
            existing_turn = await maxun_turn_repo.get(conversation_id, request.maxun_turn_id)
            if existing_turn and existing_turn.request_digest == request_digest:
                replay = _stored_turn_result(existing_turn)
                if replay is not None:
                    return existing_turn, replay, False
                if recover_processing is None:
                    return existing_turn, None, False
                if not recover_processing:
                    updated_at = existing_turn.updated_at
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=UTC)
                    age = (datetime.now(UTC) - updated_at).total_seconds()
                    if age < _MAXUN_TURN_PROCESSING_TTL_SECONDS:
                        raise HTTPException(
                            status_code=409, detail={"code": "MAXUN_TURN_IN_PROGRESS"}
                        )
                existing_turn.status = "processing"
                existing_turn.result_json = None
                existing_turn.attempt = max(1, existing_turn.attempt) + 1
                existing_turn.cancel_requested_at = None
                existing_turn.started_at = None
                existing_turn.finished_at = None
                existing_turn.updated_at = datetime.now(UTC)
                await session.commit()
                return existing_turn, None, True
            raise HTTPException(status_code=409, detail={"code": "MAXUN_TURN_ID_REUSED"})
        await session.commit()
        return maxun_turn, None, False


async def _run_turn(
    conversation_id: str,
    request: MaxunTurnRequest,
    *,
    turn_record_id: str | None = None,
    recovered: bool = False,
) -> dict[str, Any]:
    if turn_record_id is None:
        maxun_turn, replay, recovered = await _ensure_turn_record(
            conversation_id, request, recover_processing=False
        )
        if replay is not None:
            return replay
    else:
        maxun_turn = None
        replay = None

    factory = _get_session_factory()
    async with factory() as session:
        repo = ConversationRepo(session)
        conversation = await repo.get(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})
        expected_engine = f"maxun:{request.workspace_id}"
        if conversation.engine_name != expected_engine:
            raise HTTPException(
                status_code=409, detail={"code": "MAXUN_CONVERSATION_BINDING_MISMATCH"}
            )
        if turn_record_id is not None:
            maxun_turn = await MaxunTurnRepo(session).get_by_id(turn_record_id)
            if maxun_turn is None:
                raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
        else:
            assert maxun_turn is not None
            maxun_turn = await MaxunTurnRepo(session).get_by_id(maxun_turn.id) or maxun_turn
        assert maxun_turn is not None
        replay = _stored_turn_result(maxun_turn)
        if replay is not None:
            return replay
        await _mark_turn_started(maxun_turn.id, maxun_turn.attempt, recovered)
        await session.refresh(maxun_turn)
        if maxun_turn.status != "processing" or await _turn_is_cancel_requested(maxun_turn.id):
            return _stored_turn_result(maxun_turn) or _cancelled_result()

        turn_record_id = maxun_turn.id
        message_repo = MessageRepo(session)
        prior_messages = await message_repo.list_for_conversation(conversation_id)
        sequence = len(prior_messages)
        if not recovered:
            session.add(
                _new_message(
                    conversation_id,
                    "TEXT",
                    "user",
                    {"text": request.question.strip()},
                    sequence,
                )
            )
            sequence += 1

        engine = None
        events: list[dict[str, Any]] = []
        successful_sql_seen = False
        cancellation_requested = False
        pending_answer_delta = ""
        last_delta_flush = time.monotonic()

        async def flush_answer_delta() -> None:
            nonlocal pending_answer_delta, last_delta_flush
            if not pending_answer_delta:
                return
            await _append_turn_event(
                turn_record_id,
                "answer.delta",
                {"text": pending_answer_delta[:_MAX_ANSWER_CHARS]},
            )
            pending_answer_delta = ""
            last_delta_flush = time.monotonic()

        try:
            engine = await resolve_engine(
                expected_engine,
                session,
                maxun_workspace_signature=request.data_signature,
                maxun_workspace_version=request.workspace_version,
            )
            _active_engines[maxun_turn.id] = engine
            _configure_maxun_turn_budget(engine)
            from analytics_agent.agent.compactor_registry import get_compactor

            history = build_history(
                prior_messages,
                request.question.strip(),
                compactor=get_compactor(),
                max_history_tokens=settings.max_history_tokens,
            )
            if os.environ.get("MOCK_LLM") == "1":
                event_stream = _mock_maxun_events(engine, conversation_id, request.question.strip())
            else:
                graph = build_graph(
                    engine_name=expected_engine,
                    engine=engine,
                    engine_tools=engine.get_tools(),
                    context_tools=[],
                    disabled_tools={"create_chart"},
                    enabled_mutations=set(),
                    maxun_readonly=True,
                )
                event_stream = stream_graph_events(
                    graph=graph,
                    user_text=request.question.strip(),
                    conversation_id=conversation_id,
                    engine_name=expected_engine,
                    keepalive_interval=settings.sse_keepalive_interval,
                    history=history,
                )
            async for event in event_stream:
                if await _turn_is_cancel_requested(maxun_turn.id):
                    cancellation_requested = True
                    cancel_active = getattr(engine, "cancel_active", None)
                    if callable(cancel_active):
                        with contextlib.suppress(Exception):
                            await cancel_active()
                    break
                event_type = event.get("event")
                if pending_answer_delta and time.monotonic() - last_delta_flush >= 0.25:
                    await flush_answer_delta()
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
                if event_type == "SQL":
                    payload = event.get("payload")
                    if isinstance(payload, dict) and isinstance(payload.get("sql"), str):
                        successful_sql_seen = True
                        await _append_turn_event(
                            maxun_turn.id,
                            "query.result",
                            {
                                "sql": payload.get("sql"),
                                "columns": payload.get("columns", []),
                                "rows": payload.get("rows", []),
                                "truncated": bool(payload.get("truncated", False)),
                            },
                        )
                elif event_type == "TEXT" and successful_sql_seen:
                    payload = event.get("payload")
                    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                        pending_answer_delta = (f"{pending_answer_delta}{payload['text']}")[
                            :_MAX_ANSWER_CHARS
                        ]
                        if len(pending_answer_delta.encode()) >= 4096:
                            await flush_answer_delta()
                # Partial Agent events belong to the durable turn-event ledger,
                # not conversation history. Only the final MAXUN_RESULT below
                # becomes prompt history, so a recovered attempt cannot feed
                # unfinished tool/prose output back into the next prompt.
            await flush_answer_delta()
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
        finally:
            _active_engines.pop(maxun_turn.id, None)
            if engine is not None:
                with contextlib.suppress(Exception):
                    await engine.aclose()

        result = _cancelled_result() if cancellation_requested else _result_from_events(events)
        if result["status"] == "error":
            result["error"] = result.get("error") or {
                "code": "MAXUN_TURN_FAILED",
                "message": "The workspace question could not be completed",
            }
        locked_result = await session.execute(
            select(MaxunTurn)
            .where(MaxunTurn.id == maxun_turn.id)
            .execution_options(populate_existing=True)
            .with_for_update()
        )
        locked_turn = locked_result.scalar_one_or_none()
        if locked_turn is None:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
        if locked_turn.status != "processing":
            existing_result = _stored_turn_result(locked_turn)
            await session.rollback()
            return existing_result or _cancelled_result()
        if locked_turn.cancel_requested_at is not None:
            result = _cancelled_result()
        locked_turn.status = result["status"]
        locked_turn.result_json = orjson.dumps(result).decode()
        locked_turn.finished_at = datetime.now(UTC)
        locked_turn.updated_at = datetime.now(UTC)
        maxun_turn = locked_turn
        terminal_event = {
            "completed": "turn.completed",
            "error": "turn.failed",
            "cancelled": "turn.cancelled",
        }.get(result["status"])
        if terminal_event:
            _append_event_in_session(
                session,
                maxun_turn,
                terminal_event,
                {
                    "status": result["status"],
                    "answer": result["answer"],
                    "sql": result["sql"],
                    "columns": result["columns"],
                    "rows": result["rows"],
                    "truncated": result["truncated"],
                    "error": result.get("error"),
                },
            )
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
            raise HTTPException(
                status_code=503, detail={"code": "MAXUN_WORKSPACE_UNAVAILABLE"}
            ) from error
        finally:
            if engine is not None:
                with contextlib.suppress(Exception):
                    await engine.aclose()


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    authorization: str | None = Header(default=None),
) -> None:
    _internal_authorized(authorization)
    if not _UUID_RE.fullmatch(conversation_id):
        raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})

    lock = _lock_for(conversation_id)
    try:
        async with lock:
            factory = _get_session_factory()
            async with factory() as session:
                repo = ConversationRepo(session)
                conversation = await repo.get(conversation_id)
                if not conversation:
                    return
                if not conversation.engine_name.startswith("maxun:"):
                    raise HTTPException(
                        status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"}
                    )
                await repo.delete(conversation_id)
    finally:
        _turn_locks.pop(conversation_id, None)


def _turn_status_payload(
    record: MaxunTurn,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "maxun_turn_id": record.maxun_turn_id,
        "status": record.status,
        "attempt": record.attempt,
        "last_event_sequence": max(0, record.next_event_sequence - 1),
        "result": result,
    }


async def _background_turn(
    conversation_id: str,
    request: MaxunTurnRequest,
    turn_record_id: str,
    recovered: bool,
) -> None:
    try:
        async with _turn_capacity:
            await _run_turn(
                conversation_id,
                request,
                turn_record_id=turn_record_id,
                recovered=recovered,
            )
    except Exception as error:
        logger.info("Maxun background turn stopped: %s", type(error).__name__)
    finally:
        current = asyncio.current_task()
        if _turn_tasks.get(turn_record_id) is current:
            _turn_tasks.pop(turn_record_id, None)


def _schedule_turn(
    conversation_id: str,
    request: MaxunTurnRequest,
    record: MaxunTurn,
    recovered: bool,
) -> None:
    existing = _turn_tasks.get(record.id)
    if existing is not None and not existing.done():
        return
    _turn_tasks[record.id] = asyncio.create_task(
        _background_turn(conversation_id, request, record.id, recovered)
    )


@router.put("/{conversation_id}/turns/{maxun_turn_id}")
async def ensure_turn(
    conversation_id: str,
    maxun_turn_id: str,
    body: MaxunTurnRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    _validate_snapshot(body.workspace_id, body.workspace_version, body.data_signature)
    if body.maxun_turn_id != maxun_turn_id or not _UUID_RE.fullmatch(maxun_turn_id):
        raise HTTPException(status_code=400, detail={"code": "MAXUN_TURN_ID_INVALID"})
    if not _UUID_RE.fullmatch(conversation_id):
        raise HTTPException(status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"})

    lock = _lock_for(conversation_id)
    try:
        async with lock:
            existing_task = _turn_tasks.get(maxun_turn_id)
            # The task key is the durable MaxunTurn record ID, not the public UUID.
            # Lookup the record first so a process restart can recover it safely.
            factory = _get_session_factory()
            async with factory() as session:
                conversation = await ConversationRepo(session).get(conversation_id)
                if not conversation:
                    raise HTTPException(
                        status_code=404, detail={"code": "MAXUN_CONVERSATION_NOT_FOUND"}
                    )
                record = await MaxunTurnRepo(session).get(conversation_id, maxun_turn_id)
            if record is not None:
                existing_task = _turn_tasks.get(record.id)
            record, replay, recovered = await _ensure_turn_record(
                conversation_id,
                body,
                recover_processing=None
                if existing_task is not None and not existing_task.done()
                else True,
            )
            if replay is not None:
                return _turn_status_payload(record, replay)
            _schedule_turn(conversation_id, body, record, recovered)
            return _turn_status_payload(record)
    finally:
        _turn_locks.pop(conversation_id, None)


@router.get("/{conversation_id}/turns/{maxun_turn_id}/events")
async def list_turn_events(
    conversation_id: str,
    maxun_turn_id: str,
    after: int = 0,
    limit: int = 100,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    if not _UUID_RE.fullmatch(conversation_id) or not _UUID_RE.fullmatch(maxun_turn_id):
        raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
    if after < 0 or after > 2_000_000_000 or limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail={"code": "MAXUN_EVENT_CURSOR_INVALID"})
    factory = _get_session_factory()
    async with factory() as session:
        record = await MaxunTurnRepo(session).get(conversation_id, maxun_turn_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
        events = await MaxunTurnRepo(session).list_events(record.id, after, limit)
        return {
            **_turn_status_payload(record, _stored_turn_result(record)),
            "events": [
                {
                    "id": event.sequence,
                    "type": event.event_type,
                    "payload": orjson.loads(event.payload),
                }
                for event in events
            ],
        }


@router.post("/{conversation_id}/turns/{maxun_turn_id}/cancel")
async def cancel_turn(
    conversation_id: str,
    maxun_turn_id: str,
    body: MaxunTurnCancelRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    _validate_snapshot(body.workspace_id, body.workspace_version, body.data_signature)
    if not _UUID_RE.fullmatch(conversation_id) or not _UUID_RE.fullmatch(maxun_turn_id):
        raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
    factory = _get_session_factory()
    result: dict[str, Any] | None = None
    record_id: str | None = None
    async with factory() as session:
        conversation = await ConversationRepo(session).get(conversation_id)
        if conversation is None or conversation.engine_name != f"maxun:{body.workspace_id}":
            raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
        locked_result = await session.execute(
            select(MaxunTurn)
            .where(
                MaxunTurn.conversation_id == conversation_id,
                MaxunTurn.maxun_turn_id == maxun_turn_id,
            )
            .with_for_update()
        )
        record = locked_result.scalar_one_or_none()
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "MAXUN_TURN_NOT_FOUND"})
        record_id = record.id
        result = _stored_turn_result(record)
        if record.status == "processing":
            result = _cancelled_result()
            now = datetime.now(UTC)
            record.cancel_requested_at = now
            record.status = "cancelled"
            record.result_json = orjson.dumps(result).decode()
            record.finished_at = now
            record.updated_at = now
            _append_event_in_session(session, record, "turn.cancelled", result)
            await session.commit()
        else:
            await session.rollback()
    active_engine = _active_engines.get(record_id)
    if active_engine is not None:
        cancel_active = getattr(active_engine, "cancel_active", None)
        if callable(cancel_active):
            with contextlib.suppress(Exception):
                await cancel_active()
    return result or _cancelled_result()


@router.post("/{conversation_id}/turns")
async def create_turn(
    conversation_id: str,
    body: MaxunTurnRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _internal_authorized(authorization)
    _validate_snapshot(body.workspace_id, body.workspace_version, body.data_signature)
    if not _UUID_RE.fullmatch(body.maxun_turn_id):
        raise HTTPException(status_code=400, detail={"code": "MAXUN_TURN_ID_INVALID"})
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
