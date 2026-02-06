from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Output

from .config import Settings, get_ml_client

LOG = logging.getLogger("submit_train_job")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def ensure_compute(ml: Any, name: str) -> None:
    try:
        ml.compute.get(name)
        LOG.info("Compute exists: %s", name)
        return
    except Exception:
        pass

    compute = AmlCompute(
        name=name,
        size=os.getenv("AML_COMPUTE_SKU", "Standard_DS11_v2"),
        min_instances=0,
        max_instances=int(os.getenv("AML_COMPUTE_MAX", "2")),
        idle_time_before_scale_down=120,
    )
    ml.compute.begin_create_or_update(compute).result()
    LOG.info("Compute created: %s", name)


def ensure_env(ml: Any, name: str) -> Environment:
    env = Environment(
        name=name,
        description="Training environment for spam demo (MLflow + scikit-learn).",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(Path("src/training/conda.yml")),
    )
    return ml.environments.create_or_update(env)


def main() -> int:
    configure_logging()
    s = Settings.from_env()
    ml = get_ml_client(s)

    compute_name = os.getenv("AML_COMPUTE_NAME", "cpu-cluster")
    ensure_compute(ml, compute_name)

    env = ensure_env(ml, "spam-train-env")

    data_version = os.getenv("SPAM_DATA_VERSION", "latest")
    data_asset = f"azureml:spam_sample:{data_version}"

    job = command(
        code=str(Path("src/training")),
        command="python train.py --data ${{inputs.data}} --output-dir ${{outputs.model_output}}",
        inputs={
            "data": Input(type=AssetTypes.URI_FILE, path=data_asset, mode="ro_mount"),
        },
        outputs={
            "model_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        environment=env,
        compute=compute_name,
        experiment_name="spam-mini-mlops",
        display_name="spam-train",
    )

    created = ml.jobs.create_or_update(job)
    LOG.info("Submitted job: %s", created.name)
    LOG.info(
        "Stream logs: az ml job stream -n %s -g %s -w %s",
        created.name,
        s.resource_group,
        s.workspace_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
