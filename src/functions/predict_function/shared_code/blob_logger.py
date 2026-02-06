from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from azure.storage.blob import BlobServiceClient


@dataclass(frozen=True)
class LogResult:
    destination: str  # blob path or local path


class PredictionLogger:
    """Write one JSON document per request.

    If AZURE_STORAGE_CONNECTION_STRING is not set, falls back to writing local files
    under ./local_blob_logs/ so the project still runs without Azurite.
    """

    def __init__(self, connection_string: Optional[str], container: str) -> None:
        self._container = container
        self._service = (
            BlobServiceClient.from_connection_string(connection_string) if connection_string else None
        )

    def write(self, record: Dict[str, Any]) -> LogResult:
        now = datetime.now(timezone.utc)
        blob_name = f"predictions/{now:%Y/%m/%d}/{now:%H%M%S}-{uuid.uuid4().hex}.json"

        if self._service is None:
            base = Path(os.getenv("LOCAL_BLOB_LOG_DIR", "local_blob_logs")) / self._container
            base.mkdir(parents=True, exist_ok=True)
            path = base / blob_name.replace("/", "_")
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            return LogResult(destination=str(path))

        container_client = self._service.get_container_client(self._container)
        try:
            container_client.create_container()
        except Exception:  # noqa: BLE001
            pass

        blob_client = self._service.get_blob_client(container=self._container, blob=blob_name)
        blob_client.upload_blob(json.dumps(record).encode("utf-8"), overwrite=True)
        return LogResult(destination=f"{self._container}/{blob_name}")
