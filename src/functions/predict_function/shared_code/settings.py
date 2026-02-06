from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionSettings:
    aml_scoring_uri: str
    aml_endpoint_key: str
    storage_connection_string: str | None
    log_container: str

    @staticmethod
    def from_env() -> FunctionSettings:
        scoring_uri = os.getenv("AML_SCORING_URI", "")
        key = os.getenv("AML_ENDPOINT_KEY", "")
        if not scoring_uri:
            raise RuntimeError("Missing AML_SCORING_URI")
        if not key:
            raise RuntimeError("Missing AML_ENDPOINT_KEY")

        return FunctionSettings(
            aml_scoring_uri=scoring_uri,
            aml_endpoint_key=key,
            storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            log_container=os.getenv("BLOB_LOG_CONTAINER", "logs"),
        )
