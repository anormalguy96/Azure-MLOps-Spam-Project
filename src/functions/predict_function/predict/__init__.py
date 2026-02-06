from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

import azure.functions as func

from shared_code.aml_client import AMLOnlineEndpointClient
from shared_code.blob_logger import PredictionLogger
from shared_code.settings import FunctionSettings

LOG = logging.getLogger("function.predict")


def _json_response(payload: Dict[str, Any], status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status,
        mimetype="application/json",
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    start = time.perf_counter()
    try:
        settings = FunctionSettings.from_env()
        body = req.get_json()
        text = str(body.get("text", "")).strip()
        if not text:
            return _json_response({"error": "Missing 'text' in JSON body"}, 400)

        client = AMLOnlineEndpointClient(settings.aml_scoring_uri, settings.aml_endpoint_key)
        pred = client.predict(text)

        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"text": text},
            "prediction": {"predictions": pred.predictions},
            "endpoint_latency_ms": pred.latency_ms,
            "function_latency_ms": int((time.perf_counter() - start) * 1000),
        }

        logger = PredictionLogger(settings.storage_connection_string, settings.log_container)
        log_res = logger.write(record)
        record["log_destination"] = log_res.destination

        return _json_response(record, 200)

    except ValueError:
        return _json_response({"error": "Invalid JSON body"}, 400)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Unhandled error")
        return _json_response({"error": "server_error", "detail": str(exc)}, 500)
