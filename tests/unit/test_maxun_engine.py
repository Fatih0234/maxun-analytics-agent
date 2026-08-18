from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from analytics_agent.engines.maxun import engine as engine_module
from analytics_agent.engines.maxun.engine import (
    MaxunQueryError,
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


def request_body(rows=None):
    rows = rows or [
        {"Price": "1.5", "Name": "A"},
        {"Price": "2.5", "Name": "B"},
    ]
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
                "sourceOrder": 0,
                "projectionId": IDS["projection"],
                "runId": IDS["run"],
                "robotId": IDS["robot"],
                "mappingId": IDS["mapping"],
                "dataFormatId": IDS["format"],
                "sourceDatasetKey": "products",
                "sourceSchemaSignature": "schema-v1",
                "displayName": "Catalog",
                "role": "own_catalog",
                "capturedAt": "2026-08-15T10:01:00Z",
                "projection": {"columns": ["Price", "Name"], "rows": rows},
            }
        ],
    }


@pytest.fixture
def workspace(tmp_path: Path):
    request = MaterializationRequest.model_validate(request_body())
    Materializer(tmp_path).materialize(request)
    return tmp_path


def test_workspace_queries_are_read_only_and_bounded(workspace: Path):
    engine = MaxunWorkspaceEngine(
        IDS["workspace"],
        root=workspace,
        expected_signature="a" * 64,
        expected_version=1,
    )
    try:
        assert engine._run_query("SELECT COUNT(*) AS n FROM data") == {
            "columns": ["n"],
            "rows": [{"n": 2}],
            "truncated": False,
        }
        assert engine._run_query('SELECT SUM("Price") AS total FROM data')["rows"] == [
            {"total": 4.0}
        ]
        assert engine._run_query("WITH x AS (SELECT * FROM data) SELECT COUNT(*) AS n FROM x")[
            "rows"
        ] == [{"n": 2}]
        aliased = engine._run_query(
            'SELECT mine."Price" AS mine_price, competitor."Price" AS competitor_price '
            'FROM data AS mine JOIN data AS competitor ON mine."Name" = competitor."Name"'
        )
        assert aliased["rows"] == [
            {"mine_price": 1.5, "competitor_price": 1.5},
            {"mine_price": 2.5, "competitor_price": 2.5},
        ]
    finally:
        asyncio.run(engine.aclose())


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM other",
        "SELECT * FROM __maxun_sources",
        "WITH __maxun_sources AS (SELECT * FROM data) SELECT * FROM __maxun_sources",
        "WITH outer_cte AS (WITH __maxun_sources AS (SELECT * FROM data) SELECT * FROM __maxun_sources) SELECT * FROM __maxun_sources",
        "WITH outer_cte AS (WITH __maxun_manifest AS (SELECT * FROM data) SELECT * FROM __maxun_manifest) SELECT * FROM __maxun_manifest",
        "SELECT duckdb_databases() FROM data",
        "SELECT read_csv('/etc/passwd') FROM data",
        "SELECT * FROM read_parquet('/tmp/private.parquet')",
        "ATTACH '/tmp/private.db' AS private",
        "INSTALL httpfs",
        "LOAD httpfs",
        "PRAGMA database_list",
        "CREATE TABLE data_copy AS SELECT * FROM data",
        "INSERT INTO data SELECT * FROM data",
        "SELECT 1",
        "SELECT * FROM data; SELECT 1",
        'SELECT unknown."Price" FROM data AS mine',
        'WITH source AS (SELECT * FROM data) SELECT source."Price" FROM source AS source',
    ],
)
def test_ast_policy_rejects_external_access_writes_and_fallbacks(sql: str):
    with pytest.raises(MaxunSQLPolicyError):
        validate_sql(sql)


def test_engine_rejects_tampered_artifact_mac(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MAXUN_REQUIRE_ARTIFACT_MAC", "true")
    monkeypatch.setenv("MAXUN_ARTIFACT_INTEGRITY_KEY", "dedicated-phase7-mac-key")
    request = MaterializationRequest.model_validate(request_body())
    Materializer(tmp_path).materialize(request)
    artifact = tmp_path / "v1" / IDS["workspace"] / "workspace.duckdb"
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(MaxunQueryError) as error:
        MaxunWorkspaceEngine(IDS["workspace"], root=tmp_path)
    assert error.value.code == "MAXUN_WORKSPACE_INTEGRITY"


def test_workspace_symlink_cannot_escape_controlled_root(workspace: Path, tmp_path: Path):
    workspace_dir = workspace / "v1" / IDS["workspace"]
    artifact = workspace_dir / "workspace.duckdb"
    workspace_dir.rename(tmp_path / "real-workspace")
    workspace_dir.symlink_to(tmp_path / "real-workspace", target_is_directory=True)
    with pytest.raises(MaxunQueryError) as error:
        MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    assert error.value.code == "MAXUN_WORKSPACE_INVALID"
    artifact.unlink(missing_ok=True)


def test_invalid_namespace_and_manifest_fail_closed(workspace: Path):
    with pytest.raises(MaxunQueryError) as error:
        MaxunWorkspaceEngine("11111111-1111-4111-8111-111111111112", root=workspace)
    assert error.value.code == "MAXUN_WORKSPACE_UNAVAILABLE"

    with pytest.raises(MaxunQueryError) as error:
        MaxunWorkspaceEngine(
            IDS["workspace"],
            root=workspace,
            expected_signature="b" * 64,
            expected_version=1,
        )
    assert error.value.code == "MAXUN_WORKSPACE_INTEGRITY"

    with pytest.raises(MaxunQueryError):
        MaxunWorkspaceEngine.from_engine_name("maxun:../etc/passwd", root=workspace)
    with pytest.raises(MaxunQueryError):
        MaxunWorkspaceEngine.from_engine_name(
            "maxun:11111111-1111-4111-8111-111111111111:extra", root=workspace
        )


def test_result_row_limit_is_deterministically_truncated(workspace: Path, monkeypatch):
    monkeypatch.setattr(engine_module, "MAX_RESULT_ROWS", 1)
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    try:
        result = engine._run_query("SELECT * FROM data ORDER BY Name")
        assert result["truncated"] is True
        assert len(result["rows"]) == 1
    finally:
        asyncio.run(engine.aclose())


def test_global_query_capacity_applies_across_engine_instances(workspace: Path, monkeypatch):
    capacity = engine_module._QueryCapacity(2)
    monkeypatch.setattr(engine_module, "_QUERY_CAPACITY", capacity)
    monkeypatch.setattr(engine_module, "MAX_QUERY_SECONDS", 1.0)
    engines = [MaxunWorkspaceEngine(IDS["workspace"], root=workspace) for _ in range(3)]
    active = 0
    maximum = 0

    def delayed(sql, holder):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        time.sleep(0.05)
        active -= 1
        return {"columns": ["n"], "rows": [{"n": 2}], "truncated": False}

    for item in engines:
        item._execute_query = delayed
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(
                executor.map(lambda item: item._run_query("SELECT COUNT(*) FROM data"), engines)
            )
        assert all(result["rows"] == [{"n": 2}] for result in results)
        assert maximum <= 2
    finally:
        for item in engines:
            asyncio.run(item.aclose())
        capacity.executor.shutdown(wait=True, cancel_futures=True)


def test_result_cell_limit_is_per_value_not_per_row(workspace: Path, monkeypatch):
    cell_root = workspace / "cell-limit"
    Materializer(cell_root).materialize(
        MaterializationRequest.model_validate(
            request_body(rows=[{"Price": "x" * 60, "Name": "y" * 60}])
        )
    )
    monkeypatch.setattr(engine_module, "MAX_CELL_BYTES", 100)
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=cell_root)
    try:
        result = engine._run_query('SELECT "Price", "Name" FROM data')
        assert result["rows"]
        assert all(len(str(value)) == 60 for row in result["rows"] for value in row.values())
    finally:
        asyncio.run(engine.aclose())


def test_timeout_returns_sanitized_error(workspace: Path, monkeypatch):
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    original = engine._execute_query

    def delayed(sql, holder):
        time.sleep(0.05)
        return original(sql, holder)

    monkeypatch.setattr(engine, "_execute_query", delayed)
    monkeypatch.setattr(engine_module, "MAX_QUERY_SECONDS", 0.001)
    try:
        result = engine._run_query("SELECT COUNT(*) FROM data")
        assert result["code"] == "MAXUN_QUERY_TIMEOUT"
        assert "/" not in result["error"]
    finally:
        asyncio.run(engine.aclose())


def test_cancel_active_interrupts_running_duckdb_connection(workspace: Path):
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)

    class FakeConnection:
        interrupted = False

        def interrupt(self):
            self.interrupted = True

    fake = FakeConnection()
    started = threading.Event()

    def delayed(sql, holder):
        holder["connection"] = fake
        started.set()
        time.sleep(0.05)
        holder["connection"] = None
        return {"columns": ["n"], "rows": [{"n": 2}], "truncated": False}

    engine._execute_query = delayed
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(engine._run_query, "SELECT COUNT(*) FROM data")
            assert started.wait(timeout=1)
            asyncio.run(engine.cancel_active())
            assert fake.interrupted is True
            assert future.result(timeout=1)["rows"] == [{"n": 2}]
    finally:
        asyncio.run(engine.aclose())


def test_source_context_is_bounded_and_uses_exact_source_order(workspace: Path):
    body = request_body()
    second = copy.deepcopy(body["sources"][0])
    second.update(
        {
            "workspaceSourceId": "88888888-8888-4888-8888-888888888888",
            "projectionId": "99999999-9999-4999-8999-999999999999",
            "runId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "robotId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "mappingId": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "sourceOrder": 1,
            "displayName": "Competitor capture",
            "role": "competitor",
            "sourceDatasetKey": "competitor-products",
            "projection": {"columns": ["Price", "Name"], "rows": [{"Price": "3.5", "Name": "A"}]},
        }
    )
    body["sources"].append(second)
    root = workspace / "multi-source"
    Materializer(root).materialize(MaterializationRequest.model_validate(body))
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=root)
    try:
        aggregate = engine._run_query(
            'SELECT "_source_order" AS source_order, COUNT(*) AS rows '
            'FROM data GROUP BY "_source_order" ORDER BY source_order'
        )
        assert aggregate["rows"] == [{"source_order": 0, "rows": 2}, {"source_order": 1, "rows": 1}]
        comparison = engine._run_query(
            'SELECT own."Name" AS product, own."Price" AS own_price, '
            'competitor."Price" AS competitor_price '
            "FROM data AS own JOIN data AS competitor "
            'ON own."Name" = competitor."Name" '
            'WHERE own."_source_order" = 0 AND competitor."_source_order" = 1'
        )
        assert comparison["rows"] == [{"product": "A", "own_price": 1.5, "competitor_price": 3.5}]
        engine.configure_turn_budget(max_query_tools=3, max_sql_executions=1)
        context_tool = next(
            tool for tool in engine.get_tools() if tool.name == "get_source_context"
        )
        context = json.loads(context_tool.invoke({}))
        assert context["sourceCount"] == 2
        assert [source["sourceOrder"] for source in context["sources"]] == [0, 1]
        assert [source["displayName"] for source in context["sources"]] == [
            "Catalog",
            "Competitor capture",
        ]
        assert "Do not fuzzy-match source names or row entities." in context["rules"]
    finally:
        asyncio.run(engine.aclose())


def test_source_context_counts_against_the_bounded_query_tool_budget(workspace: Path):
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    engine.configure_turn_budget(max_query_tools=1, max_sql_executions=1)
    try:
        context_tool = next(
            tool for tool in engine.get_tools() if tool.name == "get_source_context"
        )
        execute = next(tool for tool in engine.get_tools() if tool.name == "execute_sql")
        assert json.loads(context_tool.invoke({}))["sourceCount"] == 1
        assert "MAXUN_QUERY_LIMIT" in execute.invoke({"sql": "SELECT COUNT(*) FROM data"})
    finally:
        asyncio.run(engine.aclose())


def test_per_turn_tool_budget_limits_sql_execution(workspace: Path):
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    engine.configure_turn_budget(max_query_tools=3, max_sql_executions=1)
    try:
        execute = next(tool for tool in engine.get_tools() if tool.name == "execute_sql")
        assert '"rows":[{"count":2}]' in execute.invoke(
            {"sql": "SELECT COUNT(*) AS count FROM data"}
        )
        blocked = execute.invoke({"sql": "SELECT COUNT(*) AS count FROM data"})
        assert "MAXUN_QUERY_LIMIT" in blocked
    finally:
        asyncio.run(engine.aclose())


def test_fixed_tool_surface_exposes_only_data_relation(workspace: Path):
    engine = MaxunWorkspaceEngine(IDS["workspace"], root=workspace)
    try:
        assert {tool.name for tool in engine.get_tools()} == {
            "execute_sql",
            "get_source_context",
            "list_tables",
            "get_schema",
            "preview_table",
        }
        list_tables = next(tool for tool in engine.get_tools() if tool.name == "list_tables")
        assert list_tables.invoke({}) == '[{"name":"data","schema":null}]'
    finally:
        asyncio.run(engine.aclose())
