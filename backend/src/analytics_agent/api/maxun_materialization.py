from __future__ import annotations

import json
import logging
import math
import time
from uuid import UUID, uuid4

import anyio
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from analytics_agent.maxun.materialization import (
    MAX_MATERIALIZATION_HARD_SECONDS,
    MAX_REQUEST_BYTES,
    MaterializationError,
    MaterializationRequest,
    Materializer,
    authorize_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal/maxun/materializations", tags=["maxun-materialization"])
_materializer = Materializer()


def _error(error: MaterializationError) -> HTTPException:
    return HTTPException(
        status_code=error.status, detail={"code": error.code, "error": "materialization failed"}
    )


def _operation_id(value: str | None, *, required: bool = False) -> str:
    if value is None:
        if required:
            raise MaterializationError(
                "MATERIALIZATION_INVALID_CONTRACT", "operation id is required", 400
            )
        return str(uuid4())
    try:
        parsed = str(UUID(value))
    except (TypeError, ValueError) as error:
        raise MaterializationError(
            "MATERIALIZATION_INVALID_CONTRACT", "invalid operation id", 400
        ) from error
    if parsed != value:
        raise MaterializationError("MATERIALIZATION_INVALID_CONTRACT", "invalid operation id", 400)
    return parsed


def _deadline_at(value: str | None) -> float:
    now = time.time()
    if value is None:
        return now + MAX_MATERIALIZATION_HARD_SECONDS
    try:
        deadline_ms = int(value)
    except (TypeError, ValueError) as error:
        raise MaterializationError(
            "MATERIALIZATION_INVALID_CONTRACT", "invalid materialization deadline", 400
        ) from error
    if deadline_ms <= 0:
        raise MaterializationError(
            "MATERIALIZATION_INVALID_CONTRACT", "invalid materialization deadline", 400
        )
    return min(deadline_ms / 1000, now + MAX_MATERIALIZATION_HARD_SECONDS)


@router.put("/{workspace_id}")
async def materialize_workspace(
    workspace_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    operation_header: str | None = Header(default=None, alias="X-Maxun-Operation-Id"),
    deadline_header: str | None = Header(default=None, alias="X-Maxun-Deadline-At"),
) -> dict:
    try:
        if str(UUID(workspace_id)) != workspace_id:
            raise ValueError
        authorize_token(authorization)
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_REQUEST_BYTES:
                raise MaterializationError(
                    "MATERIALIZATION_LIMIT_EXCEEDED", "request is too large", 413
                )
            raw.extend(chunk)
        raw_bytes = bytes(raw)
        parsed = await anyio.to_thread.run_sync(
            _parse_materialization_request,
            raw_bytes,
        )
        if parsed.workspace.id != workspace_id:
            raise MaterializationError(
                "MATERIALIZATION_INVALID_CONTRACT", "workspace path mismatch"
            )
        return await anyio.to_thread.run_sync(
            _materializer.materialize,
            parsed,
            len(raw_bytes),
            _operation_id(operation_header),
            _deadline_at(deadline_header),
            abandon_on_cancel=False,
        )
    except MaterializationError as error:
        raise _error(error) from error
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise _error(
            MaterializationError(
                "MATERIALIZATION_INVALID_CONTRACT", "invalid materialization contract"
            )
        ) from error
    except Exception as error:
        logger.warning(
            "Maxun materialization failed without exposing internals: %s", type(error).__name__
        )
        raise _error(
            MaterializationError("MATERIALIZATION_UNAVAILABLE", "materialization unavailable", 503)
        ) from error


_MAX_EXACT_JSON_INTEGER = 2**53 - 1


def _parse_json_int(token: str) -> int | float:
    """Decode JSON integers with the ECMAScript Number producer semantics.

    Maxun serializes native JavaScript numbers before sending this request. A
    value such as ``1e20`` therefore arrives on the wire as the integer token
    ``100000000000000000000``. Python's default decoder would retain that as
    an arbitrary-precision ``int``, but JCS models JSON numbers as IEEE-754
    doubles. Preserve exact safe integers and round larger tokens exactly as
    the JavaScript producer already did.
    """
    value = int(token)
    if abs(value) <= _MAX_EXACT_JSON_INTEGER:
        return value
    try:
        normalized = float(token)
    except (OverflowError, ValueError) as error:
        raise ValueError("JSON number is outside the finite ECMAScript Number domain") from error
    if not math.isfinite(normalized):
        raise ValueError("JSON number is outside the finite ECMAScript Number domain")
    return normalized


def _parse_json_float(token: str) -> float:
    try:
        normalized = float(token)
    except (OverflowError, ValueError) as error:
        raise ValueError("JSON number is invalid") from error
    if not math.isfinite(normalized):
        raise ValueError("JSON number is outside the finite ECMAScript Number domain")
    return normalized


def _parse_materialization_request(raw: bytes) -> MaterializationRequest:
    return MaterializationRequest.model_validate(
        json.loads(raw, parse_int=_parse_json_int, parse_float=_parse_json_float)
    )


@router.post("/{workspace_id}/cancel", status_code=204)
async def cancel_materialization(
    workspace_id: str,
    authorization: str | None = Header(default=None),
    operation_header: str | None = Header(default=None, alias="X-Maxun-Operation-Id"),
) -> None:
    try:
        if str(UUID(workspace_id)) != workspace_id:
            raise ValueError
        authorize_token(authorization)
        await anyio.to_thread.run_sync(
            _materializer.cancel_operation,
            workspace_id,
            _operation_id(operation_header, required=True),
            abandon_on_cancel=False,
        )
    except MaterializationError as error:
        raise _error(error) from error
    except ValueError as error:
        raise _error(
            MaterializationError("MATERIALIZATION_INVALID_CONTRACT", "invalid workspace id")
        ) from error
    except Exception as error:
        logger.warning(
            "Maxun materialization cancellation failed without exposing internals: %s",
            type(error).__name__,
        )
        raise _error(
            MaterializationError(
                "MATERIALIZATION_CANCEL_NOT_CONVERGED", "cancellation unavailable", 503
            )
        ) from error


@router.delete("/{workspace_id}", status_code=204)
def delete_materialization(
    workspace_id: str,
    authorization: str | None = Header(default=None),
) -> None:
    try:
        if str(UUID(workspace_id)) != workspace_id:
            raise ValueError
        authorize_token(authorization)
        _materializer.delete(workspace_id)
    except MaterializationError as error:
        raise _error(error) from error
    except ValueError as error:
        raise _error(
            MaterializationError("MATERIALIZATION_INVALID_CONTRACT", "invalid workspace id")
        ) from error
    except Exception as error:
        logger.warning(
            "Maxun materialization cleanup failed without exposing internals: %s",
            type(error).__name__,
        )
        raise _error(
            MaterializationError("MATERIALIZATION_UNAVAILABLE", "cleanup unavailable", 503)
        ) from error
