from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests


def main() -> int:
    scoring_uri = os.getenv("AML_SCORING_URI")
    key = os.getenv("AML_ENDPOINT_KEY")
    if not scoring_uri or not key:
        print("Set AML_SCORING_URI and AML_ENDPOINT_KEY in your environment.")
        return 2

    text = " ".join(sys.argv[1:]) or "WIN a free gift card now!!!"
    body: Dict[str, Any] = {"text": text}

    start = time.perf_counter()
    r = requests.post(
        scoring_uri,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=10,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    print("Status:", r.status_code, f"({elapsed_ms} ms)")
    print(json.dumps(r.json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
