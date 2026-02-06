from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from azure.ai.ml.entities import AccountKeyConfiguration, AzureBlobDatastore, Data

from .config import Settings, get_ml_client

LOG = logging.getLogger("create_data_asset")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _extract_account_key(conn_str: str) -> str:
    parts = dict(item.split("=", 1) for item in conn_str.split(";") if "=" in item and item.strip())
    key = parts.get("AccountKey")
    if not key:
        raise RuntimeError("AccountKey not found in AZURE_STORAGE_CONNECTION_STRING")
    return key


def main() -> int:
    configure_logging()
    s = Settings.from_env()
    ml = get_ml_client(s)

    datastore_name = "ext_blob_datasets"
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or _extract_account_key(s.storage_connection_string)

    store = AzureBlobDatastore(
        name=datastore_name,
        description="External blob datastore for datasets container",
        account_name=s.storage_account_name,
        container_name=s.blob_data_container,
        protocol="https",
        credentials=AccountKeyConfiguration(account_key=account_key),
    )
    ml.datastores.create_or_update(store)
    LOG.info("Datastore ready: %s", datastore_name)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    data = Data(
        name="spam_sample",
        version=version,
        description="Tiny spam/ham dataset stored in Azure Blob (CSV).",
        type="uri_file",
        path=f"azureml://datastores/{datastore_name}/paths/spam_sample.csv",
    )
    ml.data.create_or_update(data)
    LOG.info("Data asset created: %s:%s", data.name, data.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
