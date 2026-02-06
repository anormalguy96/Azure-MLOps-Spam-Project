from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import mlflow.pyfunc
import pandas as pd

LOG = logging.getLogger("score")

_MODEL: mlflow.pyfunc.PyFuncModel | None = None


def init() -> None:
    """Azure ML calls init() once per worker process.

    The runtime provides AZUREML_MODEL_DIR: path to the registered model folder on disk.
    We register an MLflow model folder (type=mlflow_model), so mlflow.pyfunc.load_model works.
    """
    global _MODEL
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set. Are you running in Azure ML?")

    LOG.info("Loading MLflow model from %s", model_dir)
    _MODEL = mlflow.pyfunc.load_model(model_dir)
    LOG.info("Model loaded.")


def _normalize_payload(raw_data: Any) -> list[str]:
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

    if isinstance(payload, dict) and "text" in payload:
        return [str(payload["text"])]
    if isinstance(payload, dict) and "texts" in payload:
        return [str(x) for x in payload["texts"]]
    if isinstance(payload, list):
        return [str(x) for x in payload]
    raise ValueError('Expected {"text":"..."} or {"texts":[...]} or ["..."].')


def run(raw_data: Any) -> dict[str, Any]:
    if _MODEL is None:
        raise RuntimeError("Model is not loaded. init() was not called?")

    start = time.perf_counter()
    texts = _normalize_payload(raw_data)

    df = pd.DataFrame({"text": texts})
    preds = _MODEL.predict(df)

    pred_list = [int(x) for x in list(preds)]
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {"predictions": pred_list, "latency_ms": elapsed_ms}
