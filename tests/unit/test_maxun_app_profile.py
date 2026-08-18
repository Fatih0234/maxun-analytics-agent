"""Tests for the narrow Maxun-only Agent application surface."""

from __future__ import annotations

from pathlib import Path

import pytest


def _paths(app) -> set[str]:
    return {route.path for route in app.routes}


def test_maxun_profile_mounts_only_health_and_internal_routes():
    from analytics_agent.main import create_app

    app = create_app("maxun")
    paths = _paths(app)

    assert app.state.analytics_agent_profile == "maxun"
    assert "/health" in paths
    assert "/internal/maxun/conversations" in paths
    assert "/internal/maxun/materializations/{workspace_id}" in paths
    assert "/api/settings/connections" not in paths
    assert "/api/chat" not in paths
    assert "/api/version" not in paths
    assert "/api/releases" not in paths
    assert "/api/engines" not in paths
    assert "/api/me" not in paths
    assert "/api/greeting" not in paths
    assert "/docs" not in paths
    assert "/redoc" not in paths
    assert "/openapi.json" not in paths
    assert not any(path.startswith("/assets") for path in paths)
    assert not any("full_path" in path for path in paths)


async def test_maxun_profile_hides_generic_http_routes_and_keeps_internal_auth():
    import httpx
    from analytics_agent.main import create_app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app("maxun")),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.get("/api/settings/connections")).status_code == 404
        assert (await client.get("/api/version")).status_code == 404
        assert (await client.get("/docs")).status_code == 404
        response = await client.post(
            "/internal/maxun/conversations",
            json={
                "workspace_id": "11111111-1111-4111-8111-111111111111",
                "workspace_version": 1,
                "data_signature": "a" * 64,
                "title": "test",
            },
        )
        assert response.status_code == 503


def test_general_profile_retains_the_existing_application_surface():
    from analytics_agent.main import create_app

    app = create_app("general")
    paths = _paths(app)

    assert app.state.analytics_agent_profile == "general"
    assert "/api/settings/connections" in paths
    assert "/api/version" in paths
    assert "/internal/maxun/conversations" in paths
    assert "/internal/maxun/materializations/{workspace_id}" in paths


def test_maxun_profile_disables_telemetry_by_default(monkeypatch):
    from analytics_agent.config import settings
    from analytics_agent.main import _configure_maxun_profile_defaults

    monkeypatch.delenv("DATAHUB_TELEMETRY_ENABLED", raising=False)
    settings.datahub_telemetry_enabled = True
    _configure_maxun_profile_defaults()
    assert settings.datahub_telemetry_enabled is False


def test_maxun_profile_preserves_explicit_telemetry_setting(monkeypatch):
    from analytics_agent.config import settings
    from analytics_agent.main import _configure_maxun_profile_defaults

    monkeypatch.setenv("DATAHUB_TELEMETRY_ENABLED", "true")
    settings.datahub_telemetry_enabled = True
    _configure_maxun_profile_defaults()
    assert settings.datahub_telemetry_enabled is True


def test_maxun_image_declares_a_non_root_runtime():
    dockerfile = (Path(__file__).parents[2] / "docker" / "Dockerfile").read_text()
    assert "useradd --system --uid 10001" in dockerfile
    assert "USER analytics-agent" in dockerfile


def test_maxun_profile_rejects_unallowlisted_llm_destinations(monkeypatch):
    from analytics_agent.main import _validate_maxun_llm_egress

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://evil.example/v1")
    with pytest.raises(RuntimeError, match="not allowlisted"):
        _validate_maxun_llm_egress()


def test_maxun_profile_accepts_provider_defaults_and_explicit_proxy(monkeypatch):
    from analytics_agent.main import _validate_maxun_llm_egress

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    _validate_maxun_llm_egress()

    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm-proxy:8080/v1")
    monkeypatch.setenv("MAXUN_APPROVED_LLM_PROXY_URL", "http://llm-proxy:8080/v1")
    _validate_maxun_llm_egress()


def test_maxun_profile_rejects_embedded_llm_url_credentials(monkeypatch):
    from analytics_agent.main import _validate_maxun_llm_egress

    monkeypatch.setenv("OPENAI_BASE_URL", "https://user:secret@api.openai.com/v1")
    with pytest.raises(RuntimeError, match="invalid"):
        _validate_maxun_llm_egress()


def test_unknown_application_profile_fails_closed():
    from analytics_agent.main import create_app

    with pytest.raises(ValueError, match="profile"):
        create_app("other")
