"""Pure post-processing helpers for EVPropNet response maps."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_window_starts_us(path):
    """Load strictly increasing absolute window starts from a text file."""

    path = Path(path)
    starts = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            start_us = int(value)
        except ValueError as exc:
            raise ValueError("Window start must be an integer on line {}".format(line_number)) from exc
        if start_us < 0:
            raise ValueError("Window start must be non-negative")
        if starts and start_us <= starts[-1]:
            raise ValueError("Window starts must be strictly increasing")
        starts.append(start_us)
    if not starts:
        raise ValueError("Window start file is empty")
    return starts


def window_timestamp_us(window_end_us, delta_t_us, anchor):
    """Convert an iterator window end into an explicit output timestamp."""

    window_end_us = int(window_end_us)
    delta_t_us = int(delta_t_us)
    if anchor == "start":
        return window_end_us - delta_t_us
    if anchor == "center":
        return window_end_us - delta_t_us // 2
    if anchor == "end":
        return window_end_us
    raise ValueError("Unknown timestamp anchor: {}".format(anchor))


def scheduled_window_bounds(start_ts_us, delta_t_us, max_duration_us):
    """Yield complete absolute windows without imposing an SDK time grid."""

    start_ts_us = int(start_ts_us)
    delta_t_us = int(delta_t_us)
    max_duration_us = int(max_duration_us)
    if start_ts_us < 0 or delta_t_us <= 0 or max_duration_us < 0:
        raise ValueError("Invalid window schedule")
    stop_ts_us = start_ts_us + max_duration_us
    window_start_us = start_ts_us
    while window_start_us + delta_t_us <= stop_ts_us:
        yield window_start_us, window_start_us + delta_t_us
        window_start_us += delta_t_us


def input_transform(
    *, sensor_height, sensor_width, network_height, network_width, mode
):
    sensor_height = int(sensor_height)
    sensor_width = int(sensor_width)
    network_height = int(network_height)
    network_width = int(network_width)
    if min(sensor_height, sensor_width, network_height, network_width) <= 0:
        raise ValueError("Input dimensions must be positive")
    if mode == "center_crop":
        if network_height > sensor_height or network_width > sensor_width:
            raise ValueError("Center crop cannot exceed the sensor frame")
        return {
            "mode": mode,
            "offset_x": int(math.ceil((sensor_width - network_width) / 2.0)),
            "offset_y": int(math.ceil((sensor_height - network_height) / 2.0)),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
    if mode == "stretch":
        return {
            "mode": mode,
            "offset_x": 0,
            "offset_y": 0,
            "scale_x": sensor_width / float(network_width),
            "scale_y": sensor_height / float(network_height),
        }
    raise ValueError("Unknown input transform mode: {}".format(mode))


def map_bbox_to_sensor(bbox_xywh, transform):
    x, y, width, height = [float(value) for value in bbox_xywh]
    return [
        x * float(transform["scale_x"]) + float(transform["offset_x"]),
        y * float(transform["scale_y"]) + float(transform["offset_y"]),
        width * float(transform["scale_x"]),
        height * float(transform["scale_y"]),
    ]


def merge_boxes(regions, tau):
    merged = []
    used = [False] * len(regions)
    for index, region in enumerate(regions):
        if used[index]:
            continue
        x1, y1, width1, height1 = region
        x_min, y_min = x1, y1
        x_max, y_max = x1 + width1, y1 + height1
        used[index] = True
        for other_index in range(index + 1, len(regions)):
            if used[other_index]:
                continue
            x2, y2, width2, height2 = regions[other_index]
            dx = max(x1 - (x2 + width2), x2 - (x1 + width1), 0)
            dy = max(y1 - (y2 + height2), y2 - (y1 + height1), 0)
            if dx * dx + dy * dy <= tau * tau:
                used[other_index] = True
                x_min = min(x_min, x2)
                y_min = min(y_min, y2)
                x_max = max(x_max, x2 + width2)
                y_max = max(y_max, y2 + height2)
        merged.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
    return merged


def extract_scored_regions(
    response,
    *,
    response_threshold=0.0,
    min_component_area=5,
    merge_dist=100,
):
    """Extract boxes and score each box with native response-map values."""

    import cv2

    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2:
        raise ValueError("EVPropNet response must be a 2-D array")
    active = response > float(response_threshold)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8
    )
    regions = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        regions.append(
            [
                int(stats[label_index, cv2.CC_STAT_LEFT]),
                int(stats[label_index, cv2.CC_STAT_TOP]),
                int(stats[label_index, cv2.CC_STAT_WIDTH]),
                int(stats[label_index, cv2.CC_STAT_HEIGHT]),
            ]
        )

    outputs = []
    for x, y, width, height in merge_boxes(regions, float(merge_dist)):
        crop = response[y : y + height, x : x + width]
        crop_active = crop > float(response_threshold)
        values = crop[crop_active]
        outputs.append(
            {
                "bbox_xywh": [x, y, width, height],
                "response_max": float(np.max(values)) if values.size else 0.0,
                "response_mean": float(np.mean(values)) if values.size else 0.0,
                "response_area": int(values.size),
            }
        )
    return outputs
