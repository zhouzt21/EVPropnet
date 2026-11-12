"""Evaluate EVPropNet bbox-adaptation checkpoints on cached held-out windows."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from response_postprocess import extract_scored_regions, input_transform, map_bbox_to_sensor

tf.compat.v1.disable_eager_execution()


def load_rows(path: Path, partition: str) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["partition"] == partition]


def iou(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_hit(pred, gt) -> bool:
    x, y, w, h = [float(v) for v in pred]
    gx, gy, gw, gh = [float(v) for v in gt]
    cx, cy = gx + gw / 2.0, gy + gh / 2.0
    return x <= cx <= x + w and y <= cy <= y + h


def prepare_images(rows, batch_indices, height, width):
    batch = []
    for idx in batch_indices:
        image = cv2.imread(rows[idx]["input_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(rows[idx]["input_path"])
        if image.shape != (height, width):
            raise ValueError(f"Input shape mismatch: {rows[idx]['sample_id']} {image.shape}")
        batch.append(np.sign(image.astype(np.float32) - 127.0)[..., None])
    return np.stack(batch)


def run_checkpoint(sess, logits, input_ph, rows, checkpoint, thresholds, merge_distances, batch_size, height, width):
    predictions_by_config = {
        (float(threshold), float(merge_distance)): []
        for threshold in thresholds for merge_distance in merge_distances
    }
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, str(checkpoint))
    transform = input_transform(sensor_height=720, sensor_width=1280, network_height=height,
                                network_width=width, mode="stretch")
    for start in range(0, len(rows), batch_size):
        indices = list(range(start, min(start + batch_size, len(rows))))
        images = prepare_images(rows, indices, height, width)
        batch_logits = sess.run(logits, feed_dict={input_ph: images})
        batch_prob = 1.0 / (1.0 + np.exp(-np.clip(batch_logits[..., 0], -40.0, 40.0)))
        for row, response in zip((rows[i] for i in indices), batch_prob):
            for threshold in thresholds:
                for merge_distance in merge_distances:
                    regions = extract_scored_regions(response, response_threshold=float(threshold),
                                                     min_component_area=3, merge_dist=float(merge_distance))
                    preds = []
                    for region in regions:
                        box = map_bbox_to_sensor(region["bbox_xywh"], transform)
                        preds.append({"bbox_xywh": [float(v) for v in box],
                                      "score": float(region["response_max"]),
                                      "response_mean": float(region["response_mean"])})
                    preds.sort(key=lambda item: item["score"], reverse=True)
                    predictions_by_config[(float(threshold), float(merge_distance))].append({
                        "sample_id": row["sample_id"], "recording_id": row["recording_id"],
                        "condition_group": row["condition_group"], "window_index": row["window_index"],
                        "input_window_id": row["input_window_id"], "gt_bbox_xywh": row["gt_bbox_xywh_sensor"],
                        "predictions": preds,
                    })
    return predictions_by_config


def summarize(predictions) -> dict:
    n = len(predictions)
    ious_top1, ious_top3 = [], []
    center_top1, center_top3 = [], []
    unmatched_top3, num_preds, best_center_error = [], [], []
    for row in predictions:
        gt = row["gt_bbox_xywh"]
        preds = row["predictions"]
        scores = [iou(pred["bbox_xywh"], gt) for pred in preds]
        ious_top1.append(scores[0] if scores else 0.0)
        ious_top3.append(max(scores[:3], default=0.0))
        center_top1.append(float(bool(preds and center_hit(preds[0]["bbox_xywh"], gt))))
        center_top3.append(float(any(center_hit(pred["bbox_xywh"], gt) for pred in preds[:3])))
        matched = any(score >= 0.3 for score in scores[:3])
        unmatched_top3.append(max(0, min(3, len(preds)) - int(matched)))
        num_preds.append(len(preds))
        errors = []
        gx, gy, gw, gh = [float(v) for v in gt]
        gc = (gx + gw / 2, gy + gh / 2)
        for pred in preds:
            x, y, w, h = [float(v) for v in pred["bbox_xywh"]]
            pc = (x + w / 2, y + h / 2)
            errors.append(float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])))
        if errors:
            best_center_error.append(min(errors))
    return {
        "windows": n,
        "recall_iou03_top1": float(np.mean(np.asarray(ious_top1) >= 0.3)) if n else None,
        "recall_iou03_top3": float(np.mean(np.asarray(ious_top3) >= 0.3)) if n else None,
        "recall_iou05_top1": float(np.mean(np.asarray(ious_top1) >= 0.5)) if n else None,
        "recall_iou05_top3": float(np.mean(np.asarray(ious_top3) >= 0.5)) if n else None,
        "center_hit_top1": float(np.mean(center_top1)) if n else None,
        "center_hit_top3": float(np.mean(center_top3)) if n else None,
        "unmatched_predictions_top3_per_present_window": float(np.mean(unmatched_top3)) if n else None,
        "mean_predictions_per_window": float(np.mean(num_preds)) if n else None,
        "mean_best_center_error_px": float(np.mean(best_center_error)) if best_center_error else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--merge-distances", nargs="+", type=float, default=[0, 4, 8, 16, 32, 100],
                        help="Component merge grid in network pixels; select on validation only.")
    parser.add_argument("--selected-threshold", type=float, default=None,
                        help="For frozen test evaluation; do not pass during validation selection")
    parser.add_argument("--selected-merge-distance", type=float, default=None,
                        help="Merge distance frozen from validation for held-out test evaluation.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--test-description", default="frozen held-out test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.dataset_manifest, args.partition)
    if not rows:
        raise SystemExit(f"No {args.partition} samples found")
    tf.compat.v1.reset_default_graph()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, args.height, args.width, 1), name="Input")
    network = importlib.import_module("Network.ResNet").ResNet(InputPH=input_ph, InitNeurons=32, Suffix="", NumOut=1)
    logits = network.Network()
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, intra_op_parallelism_threads=args.threads,
                                    inter_op_parallelism_threads=2, allow_soft_placement=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for checkpoint in args.checkpoints:
            thresholds = [args.selected_threshold] if args.selected_threshold is not None else args.thresholds
            merge_distances = (
                [args.selected_merge_distance if args.selected_merge_distance is not None else 100.0]
                if args.selected_threshold is not None else args.merge_distances
            )
            predictions_by_config = run_checkpoint(
                sess, logits, input_ph, rows, checkpoint, thresholds, merge_distances,
                args.batch_size, args.height, args.width
            )
            for threshold in thresholds:
                for merge_distance in merge_distances:
                    predictions = predictions_by_config[(float(threshold), float(merge_distance))]
                    summary = summarize(predictions)
                    result = {
                        "checkpoint": str(checkpoint),
                        "threshold": float(threshold),
                        "merge_distance_network_pixels": float(merge_distance),
                        "partition": args.partition,
                        **summary,
                    }
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    if args.partition == "test":
                        out_path = args.output.with_name(
                            f"{args.output.stem}_{checkpoint.stem}_thr{threshold:.2f}_merge{merge_distance:g}.jsonl"
                        )
                        out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
                                            encoding="utf-8")

    if args.partition == "validation":
        # Predeclared priority: maximize top-3 IoU@0.3 recall; break ties with
        # fewer unmatched proposals, higher IoU@0.5 recall, then smaller merge distance.
        selected = max(results, key=lambda row: (
            row["recall_iou03_top3"] or 0.0,
            -(row["unmatched_predictions_top3_per_present_window"] or 0.0),
            row["recall_iou05_top3"] or 0.0,
            -row["merge_distance_network_pixels"],
        ))
        selection = {
            "schema": "eventradar.evpropnet_validation_selection.v1",
            "partition": args.partition,
            "test_partition_opened": False,
            "selection_rule": "max recall_iou03_top3; tie-break fewer unmatched predictions in top3, then higher recall_iou05_top3, then smaller merge distance",
            "candidate_thresholds": [float(v) for v in args.thresholds],
            "candidate_merge_distances_network_pixels": [float(v) for v in args.merge_distances],
            "selected": selected,
            "all_validation_results": results,
        }
        args.output.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = {"schema": "eventradar.evpropnet_frozen_test_evaluation.v1",
                  "partition": args.test_description, "selection_reused_from_validation": True,
                  "results": results}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
