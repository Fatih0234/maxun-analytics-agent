from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import duckdb
import pytest
import sqlglot
from analytics_agent.engines.maxun.engine import (
    MaxunSQLPolicyError,
    MaxunWorkspaceEngine,
    validate_sql,
)
from analytics_agent.maxun.materialization import MaterializationRequest, Materializer

IDS = {
    "workspace": "11111111-1111-4111-8111-111111111111",
    "format": "22222222-2222-4222-8222-222222222222",
    "source": "33333333-3333-4333-8333-333333333333",
    "projection": "44444444-4444-4444-8444-444444444444",
    "run": "55555555-5555-4555-8555-555555555555",
    "robot": "66666666-6666-4666-8666-666666666666",
    "mapping": "77777777-7777-4777-8777-777777777777",
}


def _request() -> MaterializationRequest:
    return MaterializationRequest.model_validate(
        {
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
                    "sourceOrder": 0,
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
                    "projection": {"columns": ["Price"], "rows": [{"Price": "1.25"}]},
                }
            ],
        }
    )


def _corpus() -> dict[str, list[dict[str, str]]]:
    path = Path(__file__).parents[1] / "fixtures" / "maxun_sql_security_corpus.json"
    return json.loads(path.read_text())


def test_sql_security_corpus_is_pinned_to_the_deployed_parser():
    corpus = _corpus()
    assert corpus["version"] == 1
    assert corpus["parser"] == "duckdb"
    assert sqlglot.__version__
    assert duckdb.__version__
    assert corpus["accepted"]
    assert corpus["rejected"]


@pytest.mark.parametrize("case", _corpus()["accepted"], ids=lambda case: case["name"])
def test_accepted_sql_corpus_passes_ast_policy(case):
    assert validate_sql(case["sql"]) == case["sql"].strip().rstrip(";").strip()


@pytest.mark.parametrize("case", _corpus()["rejected"], ids=lambda case: case["name"])
def test_rejected_sql_corpus_fails_ast_policy(case):
    with pytest.raises(MaxunSQLPolicyError):
        validate_sql(case["sql"])


def test_accepted_sql_corpus_has_no_sentinel_filesystem_or_network_side_effect(
    tmp_path: Path,
):
    request = _request()
    materializer = Materializer(tmp_path)
    materializer.materialize(request)
    marker = tmp_path / "sentinel.txt"
    marker.write_text("must remain unchanged")
    adjacent = tmp_path / "adjacent.duckdb"
    with duckdb.connect(str(adjacent)) as connection:
        connection.execute("CREATE TABLE secret (value VARCHAR)")
        connection.execute("INSERT INTO secret VALUES ('private')")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        engine = MaxunWorkspaceEngine(
            IDS["workspace"],
            root=tmp_path,
            expected_signature="a" * 64,
            expected_version=1,
        )
        try:
            for case in _corpus()["accepted"]:
                result = engine._run_query(case["sql"])
                assert "error" not in result, case["name"]
        finally:
            asyncio.run(engine.aclose())
    finally:
        listener.close()
    assert marker.read_text() == "must remain unchanged"
    assert adjacent.exists()
    with duckdb.connect(str(adjacent), read_only=True) as connection:
        assert connection.execute("SELECT * FROM secret").fetchall() == [("private",)]
