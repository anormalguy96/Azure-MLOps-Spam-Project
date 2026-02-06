from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)

from .config import Settings, get_ml_client

LOG = logging.getLogger("deploy_endpoint")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _latest_model_version(ml, model_name: str) -> str:
    versions = [m.version for m in ml.models.list(name=model_name)]
    if not versions:
        raise RuntimeError(f"No versions found for model: {model_name}")
    return max(versions, key=lambda v: int(v))


def main() -> int:
    configure_logging()
    s = Settings.from_env()
    ml = get_ml_client(s)

    endpoint_name = s.aml_endpoint_name
    deployment_name = s.aml_deployment_name

    model_name = os.getenv("AML_MODEL_NAME", "spam-detector")
    model_version = os.getenv("AML_MODEL_VERSION") or _latest_model_version(ml, model_name)

    model = ml.models.get(name=model_name, version=model_version)
    LOG.info("Using model: %s:%s", model.name, model.version)

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Spam detection managed online endpoint (demo).",
        auth_mode="key",
        tags={"project": "azure-mini-mlops"},
    )
    ml.online_endpoints.begin_create_or_update(endpoint).result()
    LOG.info("Endpoint ready: %s", endpoint_name)

    env = Environment(
        name="spam-infer-env",
        description="Inference environment for spam demo.",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(Path("src/serving/conda.yml")),
    )
    env = ml.environments.create_or_update(env)

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=str(Path("src/serving")), scoring_script="score.py"
        ),
        instance_type=os.getenv("AML_ENDPOINT_SKU", "Standard_DS3_v2"),
        instance_count=int(os.getenv("AML_ENDPOINT_INSTANCES", "1")),
    )
    ml.online_deployments.begin_create_or_update(deployment).result()
    LOG.info("Deployment ready: %s/%s", endpoint_name, deployment_name)

    endpoint = ml.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml.online_endpoints.begin_create_or_update(endpoint).result()

    scoring_uri = ml.online_endpoints.get(endpoint_name).scoring_uri
    keys = ml.online_endpoints.get_keys(endpoint_name)
    LOG.info("Scoring URI: %s", scoring_uri)
    LOG.info("Primary key (store as AML_ENDPOINT_KEY): %s", keys.primary_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
