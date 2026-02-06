from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

import mlflow.pyfunc
import pandas as pd

LOG = logging.getLogger("local_server")


class _ModelWrapper:
    def __init__(self, model_dir: Path) -> None:
        self.model = mlflow.pyfunc.load_model(str(model_dir))

    def predict(self, texts: List[str]) -> List[int]:
        df = pd.DataFrame({"text": texts})
        preds = self.model.predict(df)
        return [int(x) for x in list(preds)]


class Handler(BaseHTTPRequestHandler):
    model: _ModelWrapper

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/score":
            self._send(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw) if raw else {}
            texts: List[str]
            if isinstance(data, dict) and "text" in data:
                texts = [str(data["text"])]
            elif isinstance(data, dict) and "texts" in data:
                texts = [str(x) for x in data["texts"]]
            else:
                self._send(400, {"error": "invalid_payload", "expected": {"text": "..."}})
                return

            start = time.perf_counter()
            preds = self.model.predict(texts)
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._send(200, {"predictions": preds, "latency_ms": latency_ms})
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Request failed")
            self._send(500, {"error": "server_error", "detail": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOG.info("%s - %s", self.address_string(), format % args)


def serve(host: str, port: int, model_dir: Path) -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    server = HTTPServer((host, port), Handler)
    Handler.model = _ModelWrapper(model_dir)
    LOG.info("Local scoring server running on http://%s:%s/score", host, port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local HTTP scoring server for the MLflow model.")
    p.add_argument("--model-dir", type=Path, required=True, help="Path to the MLflow model folder")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serve(args.host, args.port, args.model_dir)
