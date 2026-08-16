from __future__ import annotations

import os
from pathlib import Path

UNKNOWN_BUILD = "unknown"
IMAGE_BUILD_REVISION_FILE = Path("/app/MAXUN_ANALYTICS_IMAGE_BUILD_SHA")


def _actual_build_revision() -> str:
    try:
        baked = IMAGE_BUILD_REVISION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        baked = ""
    return baked or os.environ.get("MAXUN_ANALYTICS_IMAGE_BUILD_SHA", "").strip() or UNKNOWN_BUILD


def build_identity() -> dict[str, str | bool | None]:
    actual = _actual_build_revision()
    expected = os.environ.get("MAXUN_EXPECTED_ANALYTICS_BUILD_SHA", "").strip() or None
    return {
        "actual": actual,
        "expected": expected,
        "matches": None if expected is None else actual == expected,
    }


def validate_build_identity() -> None:
    identity = build_identity()
    if identity["expected"] is not None and identity["actual"] != identity["expected"]:
        raise RuntimeError(
            "MAXUN_EXPECTED_ANALYTICS_BUILD_SHA does not match the image build revision"
        )
