from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.functions.predict_function.shared_code.blob_logger import PredictionLogger


def test_blob_logger_local_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_BLOB_LOG_DIR", str(tmp_path))
    logger = PredictionLogger(connection_string=None, container="logs")
    res = logger.write({"hello": "world"})
    p = Path(res.destination)
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8"))["hello"] == "world"
