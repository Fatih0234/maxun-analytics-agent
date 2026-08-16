from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest
from analytics_agent.maxun.materialization import (
    MaterializationError,
    MaterializationRequest,
    Materializer,
    _settings,
    authorize_token,
    input_digest,
)
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
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            'select "Price", "_source", "_source_order" from data'
        ).fetchone() == (1.25, "Catalog", 0)
        assert connection.execute("select count(*) from __maxun_manifest").fetchone() == (1,)
    materializer.delete(IDS["workspace"])
    assert not database.exists()


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


def test_same_workspace_concurrent_builds_are_idempotent(tmp_path: Path):
    request = MaterializationRequest.model_validate(payload())
    materializer = Materializer(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(materializer.materialize, [request, request]))
    assert results[0] == results[1]
    assert results[0]["rowCount"] == 1


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


def test_authorization_fails_closed(monkeypatch):
    monkeypatch.setenv("MAXUN_ANALYTICS_INTERNAL_TOKEN", "secret")
    with pytest.raises(MaterializationError):
        authorize_token("Bearer wrong")
    authorize_token("Bearer secret")
