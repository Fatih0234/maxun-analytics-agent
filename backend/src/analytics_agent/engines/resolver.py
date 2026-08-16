"""
Engine credential resolver.

Single responsibility: given an engine name and a DB session, return a
fully-configured QueryEngine instance ready to execute queries.

All credential resolution happens here — nothing is threaded through
graph.py, agent code, or chat.py beyond the engine object itself.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def resolve_engine(
    engine_name: str,
    session: AsyncSession,
    *,
    maxun_workspace_signature: str | None = None,
    maxun_workspace_version: int | None = None,
) -> Any:
    """Resolve a configured engine or a request-scoped Maxun workspace engine.

    ``maxun:<uuid>`` is a virtual namespace. It must never be looked up in the
    global registry or fall back to a configured connector. The optional
    snapshot metadata is supplied by Maxun's authenticated internal adapter and
    is checked against the materialization manifest before a query runs.
    """
    if isinstance(engine_name, str) and engine_name.startswith("maxun:"):
        from analytics_agent.engines.maxun.engine import MaxunQueryError, MaxunWorkspaceEngine

        if maxun_workspace_signature is None or maxun_workspace_version is None:
            raise MaxunQueryError("MAXUN_SNAPSHOT_REQUIRED", "Workspace snapshot is unavailable")
        return MaxunWorkspaceEngine.from_engine_name(
            engine_name,
            expected_signature=maxun_workspace_signature,
            expected_version=maxun_workspace_version,
        )

    # Configured engine resolution below retains the existing credential
    # priority and registry behavior for non-Maxun engines.
    # Priority: SSO, PAT, private key, password, then env vars.
    from analytics_agent.api.oauth import _decrypt
    from analytics_agent.db.repository import CredentialRepo
    from analytics_agent.engines.factory import get_registry

    # Get base engine from registry (already registered at startup)
    registry = get_registry()
    if engine_name not in registry:
        raise ValueError(f"Engine '{engine_name}' not found. Available: {list(registry.keys())}")
    base_engine = registry[engine_name]

    # Load credential from DB
    cred = await CredentialRepo(session).get(engine_name)
    if not cred:
        # No credential in DB — use env var fallback (yaml connections, OTTO_BOT etc.)
        logger.debug("[resolver] %s: no DB credential, using env var fallback", engine_name)
        return base_engine

    auth_type = cred.auth_type

    if auth_type == "sso_externalbrowser":
        sso_user = cred.username or ""
        logger.info("[resolver] %s: sso_externalbrowser user=%s", engine_name, sso_user)
        if hasattr(base_engine, "with_sso_user"):
            return base_engine.with_sso_user(sso_user)

    elif auth_type == "pat":
        try:
            token = _decrypt(cred.secret_enc) if cred.secret_enc else ""
            user = cred.username or ""
            logger.info("[resolver] %s: pat user=%s", engine_name, user)
            if token and hasattr(base_engine, "with_pat_token"):
                return base_engine.with_pat_token(token, pat_user=user)
        except Exception as e:
            logger.error("[resolver] %s: PAT decrypt failed: %s", engine_name, e)

    elif auth_type == "private_key":
        try:
            pem = _decrypt(cred.secret_enc) if cred.secret_enc else ""
            # Handle both \\n (double-escaped from env storage) and \n (single-escaped)
            pem = pem.replace("\\\\n", "\n").replace("\\n", "\n")
            user = cred.username or ""
            logger.info("[resolver] %s: private_key user=%s", engine_name, user)
            if pem and hasattr(base_engine, "with_private_key"):
                return base_engine.with_private_key(pem, user=user)
        except Exception as e:
            logger.error("[resolver] %s: private_key decrypt failed: %s", engine_name, e)

    # Credential found but unrecognised type or failed — env var fallback
    logger.warning(
        "[resolver] %s: auth_type=%s unhandled, using env fallback", engine_name, auth_type
    )
    return base_engine
