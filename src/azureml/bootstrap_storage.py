from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from .config import Settings

LOG = logging.getLogger("bootstrap_storage")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def ensure_container(service: BlobServiceClient, name: str) -> None:
    container_client = service.get_container_client(name)
    try:
        container_client.create_container()
        LOG.info("Created container: %s", name)
    except Exception:
        LOG.info("Container exists (or cannot be created): %s", name)


def upload_if_changed(
    service: BlobServiceClient, container: str, blob_name: str, path: Path
) -> None:
    bc = service.get_blob_client(container=container, blob=blob_name)
    data = path.read_bytes()

    try:
        props = bc.get_blob_properties()
        if props.size == len(data):
            LOG.info(
                "Blob already exists with same size, skipping upload: %s/%s",
                container,
                blob_name,
            )
            return
    except Exception:
        pass

    bc.upload_blob(data, overwrite=True)
    LOG.info("Uploaded %s -> %s/%s", path, container, blob_name)


def main() -> int:
    configure_logging()
    s = Settings.from_env()
    service = BlobServiceClient.from_connection_string(s.storage_connection_string)

    ensure_container(service, s.blob_data_container)
    ensure_container(service, s.blob_log_container)

    local_csv = Path("data/spam_sample.csv")
    if not local_csv.exists():
        raise RuntimeError("Expected data/spam_sample.csv in repo root. Run from repo root.")

    upload_if_changed(service, s.blob_data_container, "spam_sample.csv", local_csv)
    LOG.info("Storage bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
