from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import duckdb
import pytest
from analytics_agent.api import maxun_materialization as maxun_api
from analytics_agent.maxun import materialization as materialization_module
from analytics_agent.maxun.materialization import (
    MaterializationError,
    MaterializationRequest,
    Materializer,
    _settings,
    authorize_token,
    input_digest,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

IDS = {
    "workspace": "11111111-1111-4111-8111-111111111111",
    "format": "22222222-2222-4222-8222-222222222222",
    "source": "33333333-3333-4333-8333-333333333333",
    "projection": "44444444-4444-4444-8444-444444444444",
    "run": "55555555-5555-4555-8555-555555555555",
    "robot": "66666666-6666-4666-8666-666666666666",
    "mapping": "77777777-7777-4777-8777-777777777777",
}


def payload(columns=None, rows=None, *, source_order=0):
    columns = columns or ["Price", "_source", "Name", "name"]
    rows = rows or [{column: ("1.25" if column == "Price" else column) for column in columns}]
    return {
        "contractVersion": 1,
        "workspace": {
            "id": IDS["workspace"],
            "version": 1,
            "dataSignature": "a" * 64,
            "dataFormatId": IDS["format"],
            "dataFormatSnapshot": {
                "id": IDS["format"],
                "name": "Catalog",
                "columns": [{"name": "Price", "type": "number"}],
            },
        },
        "sources": [
            {
                "workspaceSourceId": IDS["source"],
                "sourceOrder": source_order,
                "projectionId": IDS["projection"],
                "runId": IDS["run"],
                "robotId": IDS["robot"],
                "mappingId": IDS["mapping"],
                "dataFormatId": IDS["format"],
                "sourceDatasetKey": "products",
                "sourceSchemaSignature": "schema",
                "displayName": "Catalog",
                "role": "own_catalog",
                "capturedAt": "2026-08-15T10:01:00+00:00",
                "projection": {"columns": columns, "rows": rows},
            }
        ],
    }


def test_aggregate_cell_limit_rejects_before_materialization(monkeypatch):
    monkeypatch.setattr(materialization_module, "MAX_CELLS", 3)
    with pytest.raises(MaterializationError) as error:
        MaterializationRequest.model_validate(payload())
    assert error.value.code == "MATERIALIZATION_LIMIT_EXCEEDED"


def test_exact_row_keys_and_non_finite_values_are_rejected():
    body = payload(rows=[{"Price": "1.2"}])
    with pytest.raises(ValidationError):
        MaterializationRequest.model_validate(body)
    body = payload(
        rows=[
            {
                column: float("nan") if column == "Price" else column
                for column in ["Price", "_source", "Name", "name"]
            }
        ]
    )
    with pytest.raises(ValidationError):
        MaterializationRequest.model_validate(body)


def test_schema_skew_is_rejected():
    body = payload()
    second = dict(body["sources"][0])
    second.update(
        {
            "workspaceSourceId": "88888888-8888-4888-8888-888888888888",
            "projectionId": "99999999-9999-4999-8999-999999999999",
            "sourceOrder": 1,
        }
    )
    second["projection"] = {
        "columns": ["Price", "Other", "Name", "name"],
        "rows": [{"Price": "1", "Other": "x", "Name": "x", "name": "y"}],
    }
    body["sources"].append(second)
    with pytest.raises(MaterializationError) as error:
        MaterializationRequest.model_validate(body)
    assert error.value.code == "MATERIALIZATION_SCHEMA_MISMATCH"


def test_compatible_sources_share_one_data_relation(tmp_path: Path):
    body = payload()
    second = dict(body["sources"][0])
    second.update(
        {
            "workspaceSourceId": "88888888-8888-4888-8888-888888888888",
            "projectionId": "99999999-9999-4999-8999-999999999999",
            "sourceOrder": 1,
            "displayName": "Benchmark",
            "role": "benchmark",
            "projection": {
                "columns": ["Price", "_source", "Name", "name"],
                "rows": [{"Price": "2", "_source": "x", "Name": "a", "name": "b"}],
            },
        }
    )
    body["sources"].append(second)
    request = MaterializationRequest.model_validate(body)
    result = Materializer(tmp_path).materialize(request)
    assert result["sourceCount"] == 2
    assert result["rowCount"] == 2
    database = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "select count(distinct _workspace_source_id) from data"
        ).fetchone() == (2,)
        assert connection.execute("select count(*) from __maxun_sources").fetchone() == (2,)


def test_materialization_is_isolated_deterministic_and_idempotent(tmp_path: Path):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    first = materializer.materialize(request)
    second = materializer.materialize(request)
    assert first == second
    assert first["rowCount"] == 1
    assert [item["physicalName"] for item in first["schema"]] == [
        "Price",
        "c_001_3e0763ca",
        "Name",
        "c_003_8801b486",
    ]
    database = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    assert database.stat().st_mode & 0o777 == 0o600
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            'select "Price", "_source", "_source_order" from data'
        ).fetchone() == (1.25, "Catalog", 0)
        assert connection.execute("select count(*) from __maxun_manifest").fetchone() == (1,)
    materializer.delete(IDS["workspace"])
    assert not database.exists()


def test_empty_sources_preserve_schema_and_manifests(tmp_path: Path):
    body = payload()
    body["sources"][0]["projection"]["rows"] = []
    second = dict(body["sources"][0])
    second.update(
        {
            "workspaceSourceId": "88888888-8888-4888-8888-888888888888",
            "projectionId": "99999999-9999-4999-8999-999999999999",
            "sourceOrder": 1,
            "projection": {"columns": body["sources"][0]["projection"]["columns"], "rows": []},
        }
    )
    body["sources"].append(second)
    request = MaterializationRequest.model_validate(body)
    result = Materializer(tmp_path).materialize(request)
    assert result["sourceCount"] == 2
    assert result["rowCount"] == 0
    assert len(result["schema"]) == 4
    database = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("select count(*) from data").fetchone() == (0,)
        assert connection.execute("select row_count from __maxun_sources").fetchone() == (0,)


def test_supported_types_and_typed_blanks_are_materialized(tmp_path: Path):
    columns = ["Integer", "Flag", "Day", "Moment", "Payload", "Text", "Broken"]
    rows = [
        {
            "Integer": "7",
            "Flag": True,
            "Day": "2026-01-01",
            "Moment": "2026-01-01T10:00:00Z",
            "Payload": {"b": 2, "a": 1},
            "Text": "",
            "Broken": "1",
        },
        {
            "Integer": "",
            "Flag": "false",
            "Day": "",
            "Moment": "2026-01-02T10:00:00Z",
            "Payload": None,
            "Text": "x",
            "Broken": "oops",
        },
    ]
    body = payload(columns=columns, rows=rows)
    body["workspace"]["dataFormatSnapshot"]["columns"] = [
        {"name": "Integer", "type": "number"},
        {"name": "Flag", "type": "boolean"},
        {"name": "Day", "type": "date"},
        {"name": "Moment", "type": "datetime"},
        {"name": "Text", "type": "text"},
        {"name": "Broken", "type": "number"},
    ]
    request = MaterializationRequest.model_validate(body)
    result = Materializer(tmp_path).materialize(request)
    by_name = {item["logicalName"]: item for item in result["schema"]}
    assert by_name["Integer"]["duckdbType"] == "BIGINT"
    assert by_name["Flag"]["duckdbType"] == "BOOLEAN"
    assert by_name["Day"]["duckdbType"] == "DATE"
    assert by_name["Moment"]["duckdbType"] == "TIMESTAMPTZ"
    assert by_name["Payload"]["duckdbType"] == "VARCHAR"
    assert by_name["Broken"]["duckdbType"] == "VARCHAR"


def test_overflow_and_mixed_timestamp_fallbacks_preserve_original_text(tmp_path: Path):
    columns = ["Large", "Naive", "Mixed"]
    rows = [
        {
            "Large": "9223372036854775808",
            "Naive": "2026-01-01T10:00:00",
            "Mixed": "2026-01-01T10:00:00Z",
        },
        {
            "Large": "9223372036854775809",
            "Naive": "2026-01-02T10:00:00",
            "Mixed": "2026-01-02T10:00:00",
        },
    ]
    body = payload(columns=columns, rows=rows)
    body["workspace"]["dataFormatSnapshot"]["columns"] = [
        {"name": "Large", "type": "number"},
        {"name": "Naive", "type": "datetime"},
        {"name": "Mixed", "type": "datetime"},
    ]
    result = Materializer(tmp_path).materialize(MaterializationRequest.model_validate(body))
    by_name = {item["logicalName"]: item for item in result["schema"]}
    assert by_name["Large"]["duckdbType"] == "VARCHAR"
    assert by_name["Naive"]["duckdbType"] == "TIMESTAMP"
    assert by_name["Mixed"]["duckdbType"] == "VARCHAR"


def test_unusual_sql_identifiers_are_quoted_and_queryable(tmp_path: Path):
    columns = ["SELECT", "Product Name", 'a"b', "Prüfung", "123abc"]
    row = {column: f"value-{index}" for index, column in enumerate(columns)}
    request = MaterializationRequest.model_validate(payload(columns=columns, rows=[row]))
    Materializer(tmp_path).materialize(request)
    database = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        result = connection.execute(
            'SELECT "SELECT", "Product Name", "a""b", "Prüfung", "123abc" FROM data'
        ).fetchone()
    assert result == tuple(row[column] for column in columns)


def test_null_byte_identifier_is_deterministically_mapped(tmp_path: Path):
    columns = ["unsafe\x00name"]
    request = MaterializationRequest.model_validate(
        payload(columns=columns, rows=[{columns[0]: "value"}])
    )
    result = Materializer(tmp_path).materialize(request)
    assert result["schema"][0]["physicalName"].startswith("c_000_")


def test_digest_ignores_json_object_key_order():
    left = payload(rows=[{"Price": "1", "_source": {"b": 2, "a": 1}, "Name": "x", "name": "y"}])
    right = payload(rows=[{"Price": "1", "_source": {"a": 1, "b": 2}, "Name": "x", "name": "y"}])
    assert input_digest(MaterializationRequest.model_validate(left)) == input_digest(
        MaterializationRequest.model_validate(right)
    )


def test_untrusted_metadata_cannot_change_materialization_path(tmp_path: Path):
    body = payload()
    body["sources"][0]["sourceDatasetKey"] = "../../outside"
    body["sources"][0]["displayName"] = "/tmp/customer-name"
    body["workspace"]["dataFormatSnapshot"]["name"] = "../../format"
    Materializer(tmp_path).materialize(MaterializationRequest.model_validate(body))
    assert sorted(
        path.relative_to(tmp_path).parts for path in tmp_path.rglob("workspace.duckdb")
    ) == [("v1", IDS["workspace"], "workspace.duckdb")]


def test_different_workspaces_are_isolated(tmp_path: Path):
    first_body = payload()
    second_body = payload()
    second_body["workspace"]["id"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    second_body["workspace"]["dataSignature"] = "b" * 64
    materializer = Materializer(tmp_path)
    materializer.materialize(MaterializationRequest.model_validate(first_body))
    materializer.materialize(MaterializationRequest.model_validate(second_body))
    databases = sorted(
        path.relative_to(tmp_path).parts for path in tmp_path.rglob("workspace.duckdb")
    )
    assert databases == [
        ("v1", IDS["workspace"], "workspace.duckdb"),
        ("v1", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "workspace.duckdb"),
    ]


def test_failed_build_does_not_publish_partial_final_file(tmp_path: Path, monkeypatch):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)

    def fail_build(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(materializer, "_build", fail_build)
    with pytest.raises(RuntimeError, match="injected failure"):
        materializer.materialize(request)
    workspace_dir = tmp_path / "v1" / IDS["workspace"]
    assert not (workspace_dir / "workspace.duckdb").exists()
    assert not list(workspace_dir.glob("workspace.tmp.*.duckdb"))


def test_same_workspace_concurrent_builds_are_idempotent(tmp_path: Path):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(materializer.materialize, [request, request]))
    assert results[0] == results[1]
    assert results[0]["rowCount"] == 1


def test_delete_waits_for_build_and_removes_published_database(tmp_path: Path, monkeypatch):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    started = Event()
    original_build = materializer._build

    def delayed_build(*args, **kwargs):
        started.set()
        time.sleep(0.05)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(materializer, "_build", delayed_build)
    worker = Thread(target=materializer.materialize, args=(request,))
    worker.start()
    assert started.wait(2)
    materializer.delete(IDS["workspace"])
    worker.join(2)
    assert not (tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb").exists()


def test_ttl_cleanup_removes_only_expired_workspace(tmp_path: Path):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    materializer.materialize(request)
    database = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    os.utime(database, (100, 100))
    assert materializer.cleanup_expired(ttl_seconds=10, now=111) == 1
    assert not database.exists()


def test_integrity_conflict_is_rejected(tmp_path: Path):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    materializer.materialize(request)
    changed = payload(
        rows=[
            {
                column: ("2.5" if column == "Price" else column)
                for column in ["Price", "_source", "Name", "name"]
            }
        ]
    )
    with pytest.raises(MaterializationError) as error:
        materializer.materialize(MaterializationRequest.model_validate(changed))
    assert error.value.code == "MATERIALIZATION_INTEGRITY_MISMATCH"


def test_duckdb_external_access_and_extensions_are_disabled():
    connection = duckdb.connect(":memory:")
    _settings(connection)
    for statement in (
        "ATTACH '/tmp/external.db' AS external_db",
        "LOAD httpfs",
        "SELECT * FROM read_csv('/etc/passwd')",
    ):
        with pytest.raises(duckdb.Error):
            connection.execute(statement)
    connection.close()


def test_internal_http_route_enforces_token_and_returns_bounded_response(
    tmp_path: Path, monkeypatch
):
    app = FastAPI()
    app.include_router(maxun_api.router)
    monkeypatch.setenv("MAXUN_ANALYTICS_INTERNAL_TOKEN", "internal-secret")
    monkeypatch.setattr(maxun_api, "_materializer", Materializer(tmp_path))
    body = payload()
    with TestClient(app) as client:
        unauthorized = client.put(f"/internal/maxun/materializations/{IDS['workspace']}", json=body)
        assert unauthorized.status_code == 503
        response = client.put(
            f"/internal/maxun/materializations/{IDS['workspace']}",
            headers={"Authorization": "Bearer internal-secret"},
            json=body,
        )
    assert response.status_code == 200
    assert response.json()["relation"] == "data"
    assert "workspace.duckdb" not in response.text


def test_authorization_fails_closed(monkeypatch):
    monkeypatch.setenv("MAXUN_ANALYTICS_INTERNAL_TOKEN", "secret")
    with pytest.raises(MaterializationError):
        authorize_token("Bearer wrong")
    authorize_token("Bearer secret")
