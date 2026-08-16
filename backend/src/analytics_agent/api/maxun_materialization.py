from __future__ import annotations

import json
import logging
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from analytics_agent.maxun.materialization import (
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


@router.put("/{workspace_id}")
async def materialize_workspace(
    workspace_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    try:
        if str(UUID(workspace_id)) != workspace_id:
            raise ValueError
        authorize_token(authorization)
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise MaterializationError(
                "MATERIALIZATION_LIMIT_EXCEEDED", "request is too large", 413
            )
        payload = json.loads(raw)
        parsed = MaterializationRequest.model_validate(payload)
        if parsed.workspace.id != workspace_id:
            raise MaterializationError(
                "MATERIALIZATION_INVALID_CONTRACT", "workspace path mismatch"
            )
        return _materializer.materialize(parsed, request_bytes=len(raw))
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


@router.delete("/{workspace_id}", status_code=204)
async def delete_materialization(
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
