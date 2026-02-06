from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass(frozen=True)
class AMLPrediction:
    predictions: list[int]
    latency_ms: int


class AMLOnlineEndpointClient:
    def __init__(self, scoring_uri: str, api_key: str, timeout_s: float = 10.0) -> None:
        self._uri = scoring_uri
        self._key = api_key
        self._timeout_s = timeout_s

    def predict(self, text: str) -> AMLPrediction:
        start = time.perf_counter()
        r = requests.post(
            self._uri,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json={"text": text},
            timeout=self._timeout_s,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        r.raise_for_status()
        payload: Dict[str, Any] = r.json()
        preds = [int(x) for x in payload.get("predictions", [])]
        return AMLPrediction(predictions=preds, latency_ms=latency_ms)
