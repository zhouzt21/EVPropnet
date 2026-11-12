"""Fine-tune EVPropNet with M30T confirmed full-UAV bbox masks.

This is an explicit task adaptation: the source EVPropNet checkpoint was
trained for propeller-center heatmaps; this driver trains its response head for
full-UAV bbox occupancy and keeps recording-level partitions from the supplied
manifest.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def load_rows(manifest: Path, partition: str) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["partition"] == partition:
                rows.append(item)
    return rows


def load_batch(rows: list[dict], indices: list[int], height: int, width: int):
    images, targets, boxes = [], [], []
    for index in indices:
        row = rows[index]
        image = cv2.imread(row["input_path"], cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(row["target_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise FileNotFoundError(row["input_path"] if image is None else row["target_path"])
        if image.shape != (height, width) or target.shape != (height, width):
            raise ValueError(f"Unexpected sample geometry for {row['sample_id']}")
        images.append(np.sign(image.astype(np.float32) - 127.0)[..., None])
        targets.append((target.astype(np.float32) / 255.0)[..., None])
        boxes.append([float(v) for v in row["gt_bbox_xywh_sensor"]])
    return np.stack(images), np.stack(targets), boxes


def components(probability: np.ndarray, threshold: float, min_area: int = 3) -> list[tuple[list[int], float]]:
    active = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    output = []
    scale_x = 1280.0 / probability.shape[1]
    scale_y = 720.0 / probability.shape[0]
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area:
            continue
        mask = labels[y:y + h, x:x + w] == label
        values = probability[y:y + h, x:x + w][mask]
        box = [round(x * scale_x), round(y * scale_y), round(w * scale_x), round(h * scale_y)]
        output.append((box, float(values.mean()) if values.size else 0.0))
    output.sort(key=lambda item: item[1], reverse=True)
    return output


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate(sess, output_tensor, input_tensor, rows, batch_size, height, width, thresholds):
    scores = {threshold: {"iou03": [], "iou05": [], "center_hit": [], "n_pred": []} for threshold in thresholds}
    center_errors = []
    losses = []
    for start in range(0, len(rows), batch_size):
        idxs = list(range(start, min(start + batch_size, len(rows))))
        images, targets, boxes = load_batch(rows, idxs, height, width)
        outputs = sess.run(output_tensor, feed_dict={input_tensor: images})
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(outputs[..., 0], -40.0, 40.0)))
        binary_targets = targets[..., 0] >= 0.5
        losses.append(float(np.mean(np.square(probabilities - binary_targets.astype(np.float32)))))
        for probability, gt in zip(probabilities, boxes):
            py, px = np.unravel_index(int(np.argmax(probability)), probability.shape)
            pred_center = [(px + 0.5) * 1280.0 / width, (py + 0.5) * 720.0 / height]
            center_errors.append(float(np.hypot(pred_center[0] - (gt[0] + gt[2] / 2), pred_center[1] - (gt[1] + gt[3] / 2))))
            for threshold in thresholds:
                preds = components(probability, threshold)
                scores[threshold]["n_pred"].append(len(preds))
                top = preds[0][0] if preds else None
                best_iou = max((iou(box, gt) for box, _ in preds), default=0.0)
                center_inside = bool(top and top[0] <= gt[0] + gt[2] / 2 <= top[0] + top[2]
                                     and top[1] <= gt[1] + gt[3] / 2 <= top[1] + top[3])
                scores[threshold]["iou03"].append(float(best_iou >= 0.3))
                scores[threshold]["iou05"].append(float(best_iou >= 0.5))
                scores[threshold]["center_hit"].append(float(center_inside))
    result = {
        "pixel_mse": float(np.mean(losses)) if losses else None,
        "median_argmax_center_error_px": float(np.median(center_errors)) if center_errors else None,
        "thresholds": {},
    }
    for threshold, metrics in scores.items():
        result["thresholds"][str(threshold)] = {
            "recall_iou03": float(np.mean(metrics["iou03"])) if metrics["iou03"] else None,
            "recall_iou05": float(np.mean(metrics["iou05"])) if metrics["iou05"] else None,
            "center_hit": float(np.mean(metrics["center_hit"])) if metrics["center_hit"] else None,
            "mean_candidates": float(np.mean(metrics["n_pred"])) if metrics["n_pred"] else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--positive-weight", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--network-height", type=int, default=480)
    parser.add_argument("--network-width", type=int, default=480)
    parser.add_argument("--validation-stride", type=int, default=1)
    parser.add_argument("--run-label", default="m30t_bbox",
                        help="Short checkpoint label, e.g. lx_longrange_bbox")
    parser.add_argument("--train-description", default="speed-family training recordings")
    parser.add_argument("--validation-description", default="degree-family validation recordings")
    parser.add_argument("--test-description", default="rotation-family holdout; not loaded during training")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    tf.compat.v1.reset_default_graph()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.dataset_manifest, "train")
    validation_rows = load_rows(args.dataset_manifest, "validation")
    if args.validation_stride > 1:
        validation_rows = validation_rows[::args.validation_stride]
    if not train_rows or not validation_rows:
        raise SystemExit(f"Missing train/validation samples: train={len(train_rows)}, val={len(validation_rows)}")

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, args.network_height, args.network_width, 1), name="Input")
    labels_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, args.network_height, args.network_width, 1), name="BBoxMask")
    network_module = importlib.import_module("Network.ResNet")
    network = network_module.ResNet(InputPH=input_ph, InitNeurons=32, Suffix="", NumOut=1)
    logits = network.Network()
    raw_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_ph, logits=logits)
    weights = 1.0 + (args.positive_weight - 1.0) * labels_ph
    loss = tf.reduce_mean(raw_ce * weights, name="weighted_bbox_mask_bce")
    optimizer = tf.compat.v1.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)
    # Restore model/normalization state only. The published checkpoint omits
    # Adam slots, while this training graph creates them before the restore op.
    checkpoint_names = {name for name, _ in tf.train.list_variables(str(args.initial_checkpoint))}
    model_vars = [
        variable for variable in tf.compat.v1.global_variables()
        if not variable.op.name.endswith("/Adam")
        and not variable.op.name.endswith("/Adam_1")
        and variable.op.name not in {"beta1_power", "beta2_power"}
    ]
    missing = [variable.op.name for variable in model_vars if variable.op.name not in checkpoint_names]
    if missing:
        raise RuntimeError(f"Initial EVPropNet checkpoint is missing model variables: {missing[:12]}")
    saver = tf.compat.v1.train.Saver(var_list=model_vars, max_to_keep=args.epochs + 1)

    config = tf.compat.v1.ConfigProto(
        device_count={"GPU": 0},
        intra_op_parallelism_threads=args.threads,
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
    )
    thresholds = [0.25, 0.5, 0.75]
    history = []
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, str(args.initial_checkpoint))
        print(json.dumps({"loaded_checkpoint": str(args.initial_checkpoint), "train_samples": len(train_rows), "validation_samples": len(validation_rows), "runtime": "TensorFlow CPU", "shape": [args.network_height, args.network_width]}, ensure_ascii=False), flush=True)
        best_metric = -1.0
        best_checkpoint = None
        for epoch in range(args.epochs):
            started = time.perf_counter()
            order = np.random.permutation(len(train_rows)).tolist()
            train_losses = []
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                images, targets, _ = load_batch(train_rows, indices, args.network_height, args.network_width)
                _, value = sess.run([train_op, loss], feed_dict={input_ph: images, labels_ph: targets})
                train_losses.append(float(value))
            validation = evaluate(sess, logits, input_ph, validation_rows, args.batch_size,
                                  args.network_height, args.network_width, thresholds)
            # Threshold 0.5 is frozen from weighted BCE semantics; use its IoU recall to select epoch.
            val_recall = validation["thresholds"]["0.5"]["recall_iou03"] or 0.0
            checkpoint = args.output_dir / f"evpropnet_{args.run_label}_epoch_{epoch + 1}.ckpt"
            saver.save(sess, str(checkpoint))
            item = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "validation": validation,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint),
            }
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if val_recall > best_metric:
                best_metric = val_recall
                best_checkpoint = str(checkpoint)

    result = {
        "schema": "eventradar.evpropnet_bbox_training.v1",
        "dataset_manifest": str(args.dataset_manifest),
        "initial_checkpoint": str(args.initial_checkpoint),
        "objective": "weighted sigmoid BCE on full-UAV bbox occupancy masks",
        "partitions": {
            "train": args.train_description,
            "validation": args.validation_description,
            "test": args.test_description,
        },
        "test_usage": "test partition is not loaded during training, epoch selection, or threshold tuning",
        "threshold_semantics": "sigmoid(logits) >= 0.5 for frozen validation checkpoint selection; 0.25/0.75 are diagnostic only",
        "pixel_exclusion_xy": [210, 386],
        "runtime": "TensorFlow CPU",
        "config": vars(args) | {"dataset_manifest": str(args.dataset_manifest), "initial_checkpoint": str(args.initial_checkpoint), "output_dir": str(args.output_dir)},
        "best_checkpoint": best_checkpoint,
        "best_validation_recall_iou03_at_0_5": best_metric,
        "history": history,
    }
    result_path = args.output_dir / "training_metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics_path": str(result_path), "best_checkpoint": best_checkpoint, "best_validation_recall_iou03_at_0_5": best_metric}, ensure_ascii=False))


if __name__ == "__main__":
    main()
