from __future__ import annotations

import argparse
import logging
import os

from azure.ai.ml.entities import Model

from .config import Settings, get_ml_client

LOG = logging.getLogger("register_model")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register the trained MLflow model in Azure ML.")
    p.add_argument("--job-name", required=True, help="Azure ML job name from submit_train_job.py")
    p.add_argument("--model-name", default="spam-detector", help="Azure ML model name")
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    s = Settings.from_env()
    ml = get_ml_client(s)

    model_path = f"azureml://jobs/{args.job_name}/outputs/model_output"

    model = Model(
        name=args.model_name,
        description="MLflow model for spam detection (tfidf + logistic regression).",
        type="mlflow_model",
        path=model_path,
    )

    created = ml.models.create_or_update(model)
    LOG.info("Model registered: %s:%s", created.name, created.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
