from __future__ import annotations

from pathlib import Path

import pytest
from analytics_agent.engines import resolver
from analytics_agent.engines.maxun.engine import MaxunQueryError, MaxunWorkspaceEngine
from analytics_agent.maxun.materialization import MaterializationRequest, Materializer

from tests.unit.test_maxun_engine import IDS, request_body


@pytest.mark.asyncio
async def test_maxun_namespace_bypasses_global_registry(tmp_path: Path, monkeypatch):
    Materializer(tmp_path).materialize(MaterializationRequest.model_validate(request_body()))
    monkeypatch.setenv("MAXUN_MATERIALIZATION_ROOT", str(tmp_path))

    def fail_registry():
        raise AssertionError("Maxun workspaces must not use the global registry")

    monkeypatch.setattr("analytics_agent.engines.factory.get_registry", fail_registry)
    engine = await resolver.resolve_engine(
        f"maxun:{IDS['workspace']}",
        None,  # type: ignore[arg-type]
        maxun_workspace_signature="a" * 64,
        maxun_workspace_version=1,
    )
    assert isinstance(engine, MaxunWorkspaceEngine)
    await engine.aclose()


@pytest.mark.asyncio
async def test_maxun_resolution_requires_snapshot_metadata():
    with pytest.raises(MaxunQueryError) as error:
        await resolver.resolve_engine("maxun:11111111-1111-4111-8111-111111111111", None)  # type: ignore[arg-type]
    assert error.value.code == "MAXUN_SNAPSHOT_REQUIRED"


@pytest.mark.asyncio
async def test_invalid_maxun_namespace_never_falls_back(monkeypatch):
    monkeypatch.setattr("analytics_agent.engines.factory.get_registry", lambda: {"default": object()})
    with pytest.raises(MaxunQueryError) as error:
        await resolver.resolve_engine(
            "maxun:not-a-uuid",
            None,  # type: ignore[arg-type]
            maxun_workspace_signature="a" * 64,
            maxun_workspace_version=1,
        )
    assert error.value.code == "MAXUN_ENGINE_INVALID"
