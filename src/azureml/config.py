from __future__ import annotations

import os
from dataclasses import dataclass

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


@dataclass(frozen=True)
class Settings:
    subscription_id: str
    resource_group: str
    location: str
    workspace_name: str

    storage_account_name: str
    storage_connection_string: str

    blob_data_container: str
    blob_log_container: str

    aml_endpoint_name: str
    aml_deployment_name: str

    @staticmethod
    def from_env() -> "Settings":
        def req(name: str) -> str:
            val = os.getenv(name)
            if not val:
                raise RuntimeError(f"Missing required env var: {name}")
            return val

        return Settings(
            subscription_id=req("AZURE_SUBSCRIPTION_ID"),
            resource_group=req("AZURE_RESOURCE_GROUP"),
            location=os.getenv("AZURE_LOCATION", "westeurope"),
            workspace_name=req("AZUREML_WORKSPACE_NAME"),
            storage_account_name=req("STORAGE_ACCOUNT_NAME"),
            storage_connection_string=req("AZURE_STORAGE_CONNECTION_STRING"),
            blob_data_container=os.getenv("BLOB_DATA_CONTAINER", "datasets"),
            blob_log_container=os.getenv("BLOB_LOG_CONTAINER", "logs"),
            aml_endpoint_name=os.getenv("AML_ENDPOINT_NAME", "spam-endpoint"),
            aml_deployment_name=os.getenv("AML_DEPLOYMENT_NAME", "blue"),
        )


def get_ml_client(s: Settings) -> MLClient:
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient(credential, s.subscription_id, s.resource_group, s.workspace_name)
