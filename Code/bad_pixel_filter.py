"""Load and apply EventRadar's versioned bad-pixel mask."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(payload):
    unsigned = {key: value for key, value in payload.items() if key != "mask_sha256"}
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_bad_pixel_mask(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported bad-pixel mask schema")
    if str(payload.get("mask_sha256", "")) != _hash(payload):
        raise ValueError("Bad-pixel mask hash mismatch")
    payload["path"] = str(path)
    payload["pixel_coordinates"] = [
        (int(pixel["x"]), int(pixel["y"])) for pixel in payload.get("pixels", [])
    ]
    return payload


def filter_events(events, pixels, *, width):
    if events.size == 0 or not pixels:
        return events
    ids = events["y"].astype(np.int64) * int(width) + events["x"].astype(np.int64)
    masked = np.asarray(
        [int(y) * int(width) + int(x) for x, y in pixels], dtype=np.int64
    )
    return events[~np.isin(ids, masked)]
