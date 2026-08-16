from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SOURCES = 100
MAX_PROJECTION_ROWS = 10_000
MAX_PROJECTION_COLUMNS = 500
MAX_PROJECTION_BYTES = 25 * 1024 * 1024
MAX_ROWS = 50_000
# The measured safe Phase 3 envelope is one million cells. Larger shapes can be
# revisited with a streaming transport instead of increasing in-memory JSON caps.
MAX_CELLS = 1_000_000
MAX_CELL_BYTES = 1_048_576
MAX_REQUEST_BYTES = 128 * 1024 * 1024
RESERVED_COLUMNS = {
    "_source",
    "_source_role",
    "_captured_at",
    "_captured_at_ts",
    "_run_id",
    "_robot_id",
    "_mapping_id",
    "_projection_id",
    "_source_dataset_key",
    "_workspace_source_id",
    "_source_order",
}
SYSTEM_COLUMNS = [
    ("_source", "VARCHAR"),
    ("_source_role", "VARCHAR"),
    ("_captured_at", "VARCHAR"),
    ("_captured_at_ts", "TIMESTAMPTZ"),
    ("_run_id", "VARCHAR"),
    ("_robot_id", "VARCHAR"),
    ("_mapping_id", "VARCHAR"),
    ("_projection_id", "VARCHAR"),
    ("_source_dataset_key", "VARCHAR"),
    ("_workspace_source_id", "VARCHAR"),
    ("_source_order", "INTEGER"),
]


class MaterializationError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


class Projection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str]
    rows: list[dict[str, Any]]

    @field_validator("columns")
    @classmethod
    def columns_are_strings(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) for item in value):
            raise ValueError("projection columns must be safe strings")
        if len(set(value)) != len(value):
            raise ValueError("duplicate projection column")
        return value


class WorkspaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    columns: list[dict[str, Any]] = Field(default_factory=list)
    sourceFileName: str | None = None


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspaceSourceId: str
    sourceOrder: int
    projectionId: str
    runId: str
    robotId: str
    mappingId: str
    dataFormatId: str
    sourceDatasetKey: str
    sourceSchemaSignature: str
    displayName: str
    role: str
    capturedAt: str
    projection: Projection


class Workspace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: int
    dataSignature: str
    dataFormatId: str
    dataFormatSnapshot: WorkspaceSnapshot


class MaterializationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contractVersion: int
    workspace: Workspace
    sources: list[Source]

    @model_validator(mode="after")
    def validate_contract(self) -> MaterializationRequest:
        _uuid(self.workspace.id)
        _uuid(self.workspace.dataFormatId)
        if self.workspace.dataFormatSnapshot.id != self.workspace.dataFormatId:
            raise ValueError("data format snapshot mismatch")
        if self.contractVersion != 1 or self.workspace.version < 1:
            raise ValueError("unsupported contract")
        if not re.fullmatch(r"[0-9a-f]{64}", self.workspace.dataSignature):
            raise ValueError("invalid data signature")
        if not self.sources or len(self.sources) > MAX_SOURCES:
            raise ValueError("invalid source count")
        orders = [s.sourceOrder for s in self.sources]
        if sorted(orders) != list(range(len(self.sources))):
            raise ValueError("source orders must be contiguous")
        if len({s.workspaceSourceId for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate workspace source")
        if len({s.projectionId for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate projection")
        expected_columns: list[str] | None = None
        rows = cells = 0
        for source in sorted(self.sources, key=lambda item: item.sourceOrder):
            for value in (
                source.workspaceSourceId,
                source.projectionId,
                source.runId,
                source.robotId,
                source.mappingId,
                source.dataFormatId,
            ):
                _uuid(value)
            if source.dataFormatId != self.workspace.dataFormatId:
                raise ValueError("source data format mismatch")
            columns = source.projection.columns
            if (
                len(columns) > MAX_PROJECTION_COLUMNS
                or len(source.projection.rows) > MAX_PROJECTION_ROWS
            ):
                raise MaterializationError(
                    "MATERIALIZATION_LIMIT_EXCEEDED", "projection limits exceeded", 413
                )
            if (
                len(
                    canonical_json({"columns": columns, "rows": source.projection.rows}).encode(
                        "utf-8"
                    )
                )
                > MAX_PROJECTION_BYTES
            ):
                raise MaterializationError(
                    "MATERIALIZATION_LIMIT_EXCEEDED", "projection payload is too large", 413
                )
            if expected_columns is None:
                expected_columns = columns
            elif columns != expected_columns:
                raise MaterializationError(
                    "MATERIALIZATION_SCHEMA_MISMATCH", "source schemas differ", 409
                )
            for row in source.projection.rows:
                if set(row) != set(columns):
                    raise ValueError("row keys do not match columns")
                _reject_nonfinite(row)
                if any(
                    len(canonical_json(row[column]).encode("utf-8")) > MAX_CELL_BYTES
                    for column in columns
                ):
                    raise MaterializationError(
                        "MATERIALIZATION_LIMIT_EXCEEDED", "a materialization cell is too large", 413
                    )
            rows += len(source.projection.rows)
            cells += len(source.projection.rows) * len(columns)
        if rows > MAX_ROWS or cells > MAX_CELLS:
            raise MaterializationError(
                "MATERIALIZATION_LIMIT_EXCEEDED", "materialization limits exceeded", 413
            )
        return self


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("invalid UUID") from exc


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def input_digest(request: MaterializationRequest) -> str:
    sources = []
    for source in sorted(request.sources, key=lambda item: item.sourceOrder):
        rows = [
            {column: row[column] for column in source.projection.columns}
            for row in source.projection.rows
        ]
        sources.append(
            {
                "workspaceSourceId": source.workspaceSourceId,
                "sourceOrder": source.sourceOrder,
                "projectionId": source.projectionId,
                "runId": source.runId,
                "robotId": source.robotId,
                "mappingId": source.mappingId,
                "dataFormatId": source.dataFormatId,
                "sourceDatasetKey": source.sourceDatasetKey,
                "sourceSchemaSignature": source.sourceSchemaSignature,
                "displayName": source.displayName,
                "role": source.role,
                "capturedAt": source.capturedAt,
                "columns": source.projection.columns,
                "rows": rows,
            }
        )
    payload = {
        "contractVersion": request.contractVersion,
        "workspace": request.workspace.model_dump(),
        "sources": sources,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _hint(snapshot: WorkspaceSnapshot, name: str) -> str:
    for column in snapshot.columns:
        if column.get("name") == name:
            value = column.get("type")
            return str(value).lower() if value else ""
    return ""


def _json_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value)


def _strict_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        result = Decimal(str(value).strip()) if isinstance(value, str) else Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError
    if not result.is_finite():
        raise ValueError
    return result


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError
    return date.fromisoformat(value)


def _type_and_rule(values: list[Any], hint: str) -> tuple[str, str]:
    nonnull = [value for value in values if value is not None and value != ""]
    if hint in {
        "text",
        "currency",
        "distance",
        "url",
        "image_url",
        "country",
        "status",
        "condition",
    }:
        return "VARCHAR", "text"
    if hint in {"number", "numeric", "float", "decimal", "price"}:
        try:
            nums = [_strict_number(value) for value in nonnull]
            if all(n is not None and n == n.to_integral_value() for n in nums):
                if all(-(2**63) <= n < 2**63 for n in nums):
                    return "BIGINT", "strict-number"
                return "VARCHAR", "number-overflow-varchar"
            if all(n is not None for n in nums):
                return "DOUBLE", "strict-number"
        except ValueError:
            pass
        return "VARCHAR", "number-fallback-varchar"
    if hint == "year":
        if all(
            isinstance(v, int)
            and not isinstance(v, bool)
            and 0 <= v <= 9999
            or isinstance(v, str)
            and re.fullmatch(r"\d{4}", v)
            for v in nonnull
        ):
            return "BIGINT", "year"
        return "VARCHAR", "year-fallback-varchar"
    if hint in {"boolean", "bool"}:
        if all(
            isinstance(v, bool) or isinstance(v, str) and v.lower() in {"true", "false"}
            for v in nonnull
        ):
            return "BOOLEAN", "strict-boolean"
        return "VARCHAR", "boolean-fallback-varchar"
    if hint in {"date"}:
        try:
            [_parse_date(v) for v in nonnull]
            return "DATE", "strict-date"
        except (ValueError, TypeError):
            return "VARCHAR", "date-fallback-varchar"
    if hint in {"datetime", "timestamp", "time"}:
        try:
            parsed = [_parse_timestamp(v) for v in nonnull]
            aware = [v.tzinfo is not None for v in parsed if v is not None]
            return (
                "TIMESTAMPTZ" if all(aware) else "TIMESTAMP" if not any(aware) else "VARCHAR"
            ), "strict-timestamp"
        except (ValueError, TypeError):
            return "VARCHAR", "timestamp-fallback-varchar"
    if nonnull and all(isinstance(v, bool) for v in nonnull):
        return "BOOLEAN", "inferred-boolean"
    if nonnull and all(isinstance(v, int) and not isinstance(v, bool) for v in nonnull):
        if all(-(2**63) <= v < 2**63 for v in nonnull):
            return "BIGINT", "inferred-integer"
        return "VARCHAR", "integer-overflow-varchar"
    if nonnull and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in nonnull
    ):
        return "DOUBLE", "inferred-number"
    return "VARCHAR", "text"


def _convert(value: Any, dtype: str, rule: str) -> Any:
    if value is None:
        return None
    if dtype == "VARCHAR":
        return _json_text(value)
    if value == "":
        return None
    if dtype == "BIGINT":
        if rule == "year":
            return int(value)
        number = _strict_number(value)
        if number is None or number != number.to_integral_value() or not -(2**63) <= number < 2**63:
            raise ValueError
        return int(number)
    if dtype == "DOUBLE":
        number = _strict_number(value)
        if number is None:
            raise ValueError
        result = float(number)
        if not math.isfinite(result):
            raise ValueError
        return result
    if dtype == "BOOLEAN":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError
    if dtype == "DATE":
        return _parse_date(value)
    if dtype in {"TIMESTAMP", "TIMESTAMPTZ"}:
        parsed = _parse_timestamp(value)
        if (
            parsed is None
            or dtype == "TIMESTAMP"
            and parsed.tzinfo is not None
            or dtype == "TIMESTAMPTZ"
            and parsed.tzinfo is None
        ):
            raise ValueError
        return parsed
    raise ValueError


def _physical_names(columns: list[str]) -> list[tuple[str, str, bool]]:
    seen: set[str] = set()
    result = []
    for ordinal, logical in enumerate(columns):
        collision = (
            logical in RESERVED_COLUMNS
            or logical.lower().startswith("__maxun_")
            or logical.lower() in seen
        )
        if collision or "\x00" in logical:
            suffix = hashlib.sha256(f"{ordinal}:{logical}".encode()).hexdigest()[:8]
            physical, mapped = f"c_{ordinal:03d}_{suffix}", True
        else:
            physical, mapped = logical, False
        if physical.lower() in seen:
            raise MaterializationError(
                "MATERIALIZATION_INVALID_CONTRACT", "ambiguous customer identifier"
            )
        seen.add(physical.lower())
        result.append((logical, physical, mapped))
    return result


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    import fcntl

    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    with path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _settings(connection: duckdb.DuckDBPyConnection) -> None:
    for statement in (
        "SET enable_external_access = false",
        "SET allow_community_extensions = false",
        "SET autoinstall_known_extensions = false",
        "SET autoload_known_extensions = false",
        "SET threads = 2",
        "SET memory_limit = '512MB'",
        "SET max_temp_directory_size = '1GB'",
        "SET TimeZone = 'UTC'",
    ):
        connection.execute(statement)
    connection.execute("SET lock_configuration = true")


def _manifest(connection: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    try:
        row = connection.execute("SELECT * FROM __maxun_manifest LIMIT 1").fetchone()
    except Exception:
        return None
    if not row:
        return None
    columns = [item[0] for item in connection.description]
    return dict(zip(columns, row))


class Materializer:
    def __init__(self, root: str | Path | None = None):
        configured_root = os.environ.get("MAXUN_MATERIALIZATION_ROOT", "").strip()
        default_root = (
            Path(
                os.environ.get("ANALYTICS_AGENT_CONFIG_DIR", "~/.datahub/analytics-agent")
            ).expanduser()
            / "materializations"
        )
        self.root = Path(root or configured_root or default_root)
        try:
            concurrency = int(os.environ.get("MAXUN_MATERIALIZATION_CONCURRENCY", "2"))
        except ValueError as exc:
            raise ValueError("MAXUN_MATERIALIZATION_CONCURRENCY must be an integer") from exc
        if not 1 <= concurrency <= 16:
            raise ValueError("MAXUN_MATERIALIZATION_CONCURRENCY must be between 1 and 16")
        try:
            wait_seconds = float(os.environ.get("MAXUN_MATERIALIZATION_WAIT_SECONDS", "30"))
        except ValueError as exc:
            raise ValueError("MAXUN_MATERIALIZATION_WAIT_SECONDS must be numeric") from exc
        if not 0.1 <= wait_seconds <= 300:
            raise ValueError("MAXUN_MATERIALIZATION_WAIT_SECONDS must be between 0.1 and 300")
        self._wait_seconds = wait_seconds
        self._semaphore = threading.BoundedSemaphore(concurrency)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    @contextmanager
    def _capacity(self) -> Iterator[None]:
        if not self._semaphore.acquire(timeout=self._wait_seconds):
            raise MaterializationError(
                "MATERIALIZATION_BUSY", "materialization capacity is busy", 409
            )
        try:
            yield
        finally:
            self._semaphore.release()

    def _paths(self, workspace_id: str) -> tuple[Path, Path, Path]:
        workspace = str(_uuid(workspace_id))
        directory = self.root / "v1" / workspace
        return directory, directory / "workspace.duckdb", directory / "workspace.lock"

    def materialize(
        self, request: MaterializationRequest, request_bytes: int | None = None
    ) -> dict[str, Any]:
        if request_bytes is not None and request_bytes > MAX_REQUEST_BYTES:
            raise MaterializationError(
                "MATERIALIZATION_LIMIT_EXCEEDED", "request is too large", 413
            )
        digest = input_digest(request)
        directory, final, lock_path = self._paths(request.workspace.id)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        with self._capacity(), _lock(lock_path):
            if final.exists():
                try:
                    with duckdb.connect(str(final), read_only=True) as existing:
                        found = _manifest(existing)
                        if found and found.get("input_digest") == digest:
                            return self._response(request, digest, found, existing)
                        if found and found.get("data_signature") == request.workspace.dataSignature:
                            raise MaterializationError(
                                "MATERIALIZATION_INTEGRITY_MISMATCH",
                                "materialization input changed",
                                409,
                            )
                except MaterializationError:
                    raise
                except Exception:
                    final.unlink(missing_ok=True)
            temp = directory / f"workspace.tmp.{secrets.token_hex(12)}.duckdb"
            try:
                response = self._build(request, digest, temp)
                os.replace(temp, final)
                return response
            except Exception:
                temp.unlink(missing_ok=True)
                raise

    def _build(self, request: MaterializationRequest, digest: str, path: Path) -> dict[str, Any]:
        columns = request.sources[0].projection.columns
        physical = _physical_names(columns)
        all_values = [
            [
                source.projection.rows[i].get(column)
                for source in request.sources
                for i in range(len(source.projection.rows))
            ]
            for column in columns
        ]
        specs = []
        for ordinal, (logical, name, mapped) in enumerate(physical):
            dtype, rule = _type_and_rule(
                all_values[ordinal], _hint(request.workspace.dataFormatSnapshot, logical)
            )
            if dtype != "VARCHAR":
                try:
                    for value in all_values[ordinal]:
                        _convert(value, dtype, rule)
                except (ValueError, TypeError):
                    dtype, rule = "VARCHAR", f"{rule}-fallback-varchar"
            specs.append((ordinal, logical, name, dtype, rule, mapped))
        connection = duckdb.connect(str(path))
        try:
            _settings(connection)
            connection.execute("BEGIN TRANSACTION")
            customer_ddl = ", ".join(f"{_quote(name)} {dtype}" for _, _, name, dtype, _, _ in specs)
            system_ddl = ", ".join(f"{_quote(name)} {dtype}" for name, dtype in SYSTEM_COLUMNS)
            connection.execute(
                f"CREATE TABLE data ({customer_ddl + ', ' if customer_ddl else ''}{system_ddl})"
            )
            connection.execute(
                "CREATE TABLE __maxun_manifest (contract_version INTEGER, materialization_version INTEGER, workspace_id VARCHAR, workspace_version INTEGER, data_signature VARCHAR, input_digest VARCHAR, relation_name VARCHAR, source_count INTEGER, row_count INTEGER, column_count INTEGER, duckdb_version VARCHAR)"
            )
            connection.execute(
                "CREATE TABLE __maxun_sources (workspace_source_id VARCHAR, source_order INTEGER, display_name VARCHAR, role VARCHAR, projection_id VARCHAR, run_id VARCHAR, robot_id VARCHAR, mapping_id VARCHAR, data_format_id VARCHAR, source_dataset_key VARCHAR, source_schema_signature VARCHAR, captured_at VARCHAR, captured_at_ts TIMESTAMPTZ, row_count INTEGER)"
            )
            connection.execute(
                "CREATE TABLE __maxun_columns (ordinal INTEGER, logical_name VARCHAR, physical_name VARCHAR, declared_type VARCHAR, duckdb_type VARCHAR, conversion_rule VARCHAR, identifier_was_mapped BOOLEAN)"
            )
            for spec in specs:
                connection.execute(
                    "INSERT INTO __maxun_columns VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        spec[0],
                        spec[1],
                        spec[2],
                        _hint(request.workspace.dataFormatSnapshot, spec[1]),
                        spec[3],
                        spec[4],
                        spec[5],
                    ],
                )
            row_count = 0
            insert_names = [name for _, _, name, _, _, _ in specs] + [
                name for name, _ in SYSTEM_COLUMNS
            ]
            placeholders = ",".join("?" for _ in insert_names)
            sql = f"INSERT INTO data ({','.join(_quote(name) for name in insert_names)}) VALUES ({placeholders})"
            for source in sorted(request.sources, key=lambda item: item.sourceOrder):
                try:
                    captured = _parse_timestamp(source.capturedAt)
                except (ValueError, TypeError):
                    captured = None
                for row in source.projection.rows:
                    values = []
                    for spec in specs:
                        try:
                            values.append(_convert(row[spec[1]], spec[3], spec[4]))
                        except (ValueError, TypeError):
                            values.append(_json_text(row[spec[1]]))
                    values.extend(
                        [
                            source.displayName,
                            source.role,
                            source.capturedAt,
                            captured,
                            source.runId,
                            source.robotId,
                            source.mappingId,
                            source.projectionId,
                            source.sourceDatasetKey,
                            source.workspaceSourceId,
                            source.sourceOrder,
                        ]
                    )
                    connection.execute(sql, values)
                    row_count += 1
                connection.execute(
                    "INSERT INTO __maxun_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        source.workspaceSourceId,
                        source.sourceOrder,
                        source.displayName,
                        source.role,
                        source.projectionId,
                        source.runId,
                        source.robotId,
                        source.mappingId,
                        source.dataFormatId,
                        source.sourceDatasetKey,
                        source.sourceSchemaSignature,
                        source.capturedAt,
                        captured,
                        len(source.projection.rows),
                    ],
                )
            connection.execute(
                "INSERT INTO __maxun_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    1,
                    1,
                    request.workspace.id,
                    request.workspace.version,
                    request.workspace.dataSignature,
                    digest,
                    "data",
                    len(request.sources),
                    row_count,
                    len(specs),
                    duckdb.__version__,
                ],
            )
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        path.chmod(0o600)
        with duckdb.connect(str(path), read_only=True) as check:
            found = _manifest(check)
            if not found or found.get("input_digest") != digest:
                raise MaterializationError(
                    "MATERIALIZATION_UNAVAILABLE", "materialization verification failed", 503
                )
            return self._response(request, digest, found, check)

    def _response(
        self,
        request: MaterializationRequest,
        digest: str,
        found: dict[str, Any],
        connection: duckdb.DuckDBPyConnection,
    ) -> dict[str, Any]:
        schema = [
            {
                "logicalName": row[1],
                "physicalName": row[2],
                "duckdbType": row[4],
                "conversion": row[5],
            }
            for row in connection.execute(
                "SELECT * FROM __maxun_columns ORDER BY ordinal"
            ).fetchall()
        ]
        return {
            "contractVersion": 1,
            "materializationVersion": 1,
            "workspaceId": request.workspace.id,
            "dataSignature": request.workspace.dataSignature,
            "inputDigest": digest,
            "state": "ready",
            "relation": "data",
            "sourceCount": found["source_count"],
            "rowCount": found["row_count"],
            "schema": schema,
        }

    def delete(self, workspace_id: str) -> None:
        directory, _, lock_path = self._paths(workspace_id)
        if not directory.exists():
            return
        with _lock(lock_path):
            if directory.resolve().parent != (self.root / "v1").resolve():
                raise MaterializationError(
                    "MATERIALIZATION_INVALID_CONTRACT", "invalid materialization path"
                )
            shutil.rmtree(directory)

    def cleanup_expired(self, ttl_seconds: int = 86_400, now: float | None = None) -> int:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        base = self.root / "v1"
        if not base.exists():
            return 0
        current = now if now is not None else time.time()
        removed = 0
        for directory in base.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                workspace_id = str(UUID(directory.name))
                if directory.resolve().parent != base.resolve():
                    continue
                final = directory / "workspace.duckdb"
                if not final.exists() or current - final.stat().st_mtime <= ttl_seconds:
                    continue
                _, _, lock_path = self._paths(workspace_id)
                with _lock(lock_path):
                    if final.exists() and current - final.stat().st_mtime > ttl_seconds:
                        shutil.rmtree(directory)
                        removed += 1
            except (OSError, ValueError):
                continue
        return removed


def configured_token() -> str:
    return (
        os.environ.get("MAXUN_ANALYTICS_INTERNAL_TOKEN", "").strip()
        or os.environ.get("ANALYTICS_AGENT_INTERNAL_TOKEN", "").strip()
    )


def authorize_token(value: str | None) -> None:
    expected = configured_token()
    if (
        not expected
        or not value
        or not value.startswith("Bearer ")
        or not hmac.compare_digest(value[7:].strip(), expected)
    ):
        raise MaterializationError(
            "MATERIALIZATION_UNAVAILABLE", "materialization service unavailable", 503
        )
