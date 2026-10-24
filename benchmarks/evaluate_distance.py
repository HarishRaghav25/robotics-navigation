"""
Automated CLI Evaluation Suite for Object Detection & Geometric Distance Estimation.

Evaluates:
1. Object Detection mAP50, Precision, and Recall at IoU threshold >= 0.5.
2. Distance Estimation MAE (Mean Absolute Error), MRE (Mean Relative Error), and RMSE
   for longitudinal distance (Z) and 3D Euclidean distance against ground-truth.
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import yaml
import numpy as np
import cv2
import torch

# Ensure repository root is on sys.path for direct CLI invocation
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.detector import NavigationDetector
from geometry.distance_estimator import DistanceEstimator


def box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute VOC/COCO-style continuous Average Precision using monotonic envelope integration.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute monotonically non-increasing envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area where recall steps change
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def normalize_class_name(name: Any) -> str:
    """Normalize class name string to lower_case_with_underscores."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_dataset_paths(
    dataset_yaml_path: str,
    gt_dir_override: Optional[str] = None
) -> Tuple[Path, Path, Path, List[str]]:
    """
    Resolve dataset root directory, validation images directory, ground truth directory,
    and class names list from dataset.yaml.
    """
    yaml_file = Path(dataset_yaml_path)
    if not yaml_file.exists() and (ROOT / dataset_yaml_path).exists():
        yaml_file = ROOT / dataset_yaml_path
    yaml_file = yaml_file.resolve()
    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml_path}")

    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve dataset root path
    raw_path = cfg.get("path", ".")
    cand_root = Path(raw_path)
    if not cand_root.is_absolute():
        cand_root = (yaml_file.parent / cand_root).resolve()

    if cand_root.exists() and cand_root.is_dir():
        root_dir = cand_root
    else:
        root_dir = yaml_file.parent.resolve()

    # Resolve validation images directory
    raw_val = cfg.get("val", "val/images")
    cand_val = Path(raw_val)
    if cand_val.is_absolute() and cand_val.exists():
        val_images_dir = cand_val
    elif (root_dir / raw_val).exists():
        val_images_dir = (root_dir / raw_val).resolve()
    elif (yaml_file.parent / raw_val).exists():
        val_images_dir = (yaml_file.parent / raw_val).resolve()
    elif (root_dir / "val" / "images").exists():
        val_images_dir = (root_dir / "val" / "images").resolve()
    else:
        raise FileNotFoundError(
            f"Cannot resolve validation images directory '{raw_val}' relative to {root_dir} or {yaml_file.parent}"
        )

    # Resolve ground truth directory
    if gt_dir_override:
        gt_cand = Path(gt_dir_override)
        if not gt_cand.exists() and (ROOT / gt_dir_override).exists():
            gt_cand = ROOT / gt_dir_override
        gt_dir = gt_cand.resolve()
    else:
        raw_gt_cfg = cfg.get("ground_truth")
        cand_gt0 = (root_dir / raw_gt_cfg).resolve() if raw_gt_cfg and (root_dir / raw_gt_cfg).exists() else None
        cand_gt1 = val_images_dir.parent / "ground_truth"
        cand_gt2 = root_dir / "val" / "ground_truth"
        cand_gt3 = root_dir / "ground_truth"
        if cand_gt0 and cand_gt0.exists():
            gt_dir = cand_gt0
        elif cand_gt1.exists():
            gt_dir = cand_gt1.resolve()
        elif cand_gt2.exists():
            gt_dir = cand_gt2.resolve()
        elif cand_gt3.exists():
            gt_dir = cand_gt3.resolve()
        else:
            gt_dir = cand_gt1.resolve()

    # Resolve class names
    raw_names = cfg.get("names", {0: "cone", 1: "barrier", 2: "stop_sign"})
    if isinstance(raw_names, dict):
        class_names = [normalize_class_name(raw_names[k]) for k in sorted(raw_names.keys())]
    elif isinstance(raw_names, (list, tuple)):
        class_names = [normalize_class_name(c) for c in raw_names]
    else:
        class_names = ["cone", "barrier", "stop_sign"]

    return root_dir, val_images_dir, gt_dir, class_names


def evaluate_dataset(
    dataset_yaml: str,
    model_path: str,
    gt_dir_override: Optional[str] = None,
    camera_config: str = "configs/camera_params.yaml",
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    device: Optional[str] = None,
    distance_method: str = "hybrid",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run full automated evaluation on validation images and ground truth JSON files.

    Args:
        dataset_yaml: Path to dataset YAML configuration file.
        model_path: Path to PyTorch (.pt) or ONNX (.onnx) model.
        gt_dir_override: Optional custom ground truth directory.
        camera_config: Path to camera parameters YAML file.
        iou_thresh: IoU matching threshold (default 0.5 for mAP50).
        conf_thresh: Detection confidence threshold.
        nms_thresh: Detection NMS IoU threshold.
        device: Target execution device ("cpu", "cuda", or None for auto).
        distance_method: Geometric distance estimation method ("hybrid", "pinhole", "ground").
        verbose: If True, prints per-image inference progress.

    Returns:
        Structured evaluation metrics dictionary.
    """
    # Resolve relative paths against ROOT if not found relative to CWD
    model_p = Path(model_path)
    if not model_p.exists() and (ROOT / model_path).exists():
        model_path = str(ROOT / model_path)

    cam_p = Path(camera_config)
    if not cam_p.exists() and (ROOT / camera_config).exists():
        camera_config = str(ROOT / camera_config)

    root_dir, val_images_dir, gt_dir, class_names = resolve_dataset_paths(
        dataset_yaml, gt_dir_override=gt_dir_override
    )

    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    # Discover validation images
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in val_images_dir.iterdir() if p.suffix.lower() in valid_extensions])
    if not image_paths:
        raise ValueError(f"No validation images found in {val_images_dir}")

    print("\n" + "=" * 80)
    print("   AUTOMATED ROBOTICS DETECTION & DISTANCE ESTIMATION EVALUATION")
    print("=" * 80)
    print(f"Dataset YAML:        {dataset_yaml}")
    print(f"Validation Images:   {val_images_dir} ({len(image_paths)} images)")
    print(f"Ground Truth Dir:    {gt_dir}")
    print(f"Model Checkpoint:    {model_path}")
    print(f"Camera Config:       {camera_config}")
    print(f"IoU Threshold:       {iou_thresh}")
    print(f"Conf Threshold:      {conf_thresh}")
    print(f"NMS Threshold:       {nms_thresh}")
    print(f"Distance Method:     {distance_method}")
    print("=" * 80 + "\n")

    # Initialize Detector & Distance Estimator
    detector = NavigationDetector(
        model_path=model_path,
        backend="auto",
        conf_thresh=conf_thresh,
        iou_thresh=nms_thresh,
        device=device
    )
    dist_estimator = DistanceEstimator(config_path=camera_config)

    # Containers for predictions and ground-truths
    # All predictions: list of dicts
    all_predictions: List[Dict[str, Any]] = []
    # Ground-truths indexed by (image_stem, class_name)
    gts_by_image: Dict[str, List[Dict[str, Any]]] = {}

    start_eval_time = time.perf_counter()

    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warning] Failed to load image {img_path}. Skipping.")
            continue

        # Load Ground Truth JSON
        gt_file = gt_dir / f"{stem}.json"
        gt_objects = []
        if gt_file.exists():
            with open(gt_file, "r") as f:
                raw_gt = json.load(f)

            if isinstance(raw_gt, dict):
                raw_items = raw_gt.get("annotations", raw_gt.get("objects", raw_gt.get("labels", [raw_gt])))
            elif isinstance(raw_gt, list):
                raw_items = raw_gt
            else:
                raw_items = []
            if not isinstance(raw_items, list):
                raw_items = [raw_items]

            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                cname = normalize_class_name(item.get("class_name", ""))
                if not cname and "class_id" in item:
                    try:
                        cid = int(item["class_id"])
                        if 0 <= cid < len(class_names):
                            cname = class_names[cid]
                    except (ValueError, TypeError):
                        pass

                bbox = item.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4 or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                    continue

                z_val = float(item.get("Z", item.get("distance_z", item.get("z", item.get("depth", item.get("distance_3d", 0.0))))))
                d3d_val = float(item.get("distance_3d", item.get("dist_3d", item.get("distance", z_val))))
                gt_objects.append({
                    "class_name": cname,
                    "bbox": bbox,
                    "Z": z_val,
                    "distance_3d": d3d_val
                })
        gts_by_image[stem] = gt_objects

        # Run Model Inference
        raw_dets = detector.detect(img)
        for det in raw_dets:
            cname = normalize_class_name(det.get("class_name", ""))
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = float(det.get("confidence", 0.0))

            if len(bbox) != 4 or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                continue

            try:
                dist_info = dist_estimator.estimate_distance(
                    class_name=cname,
                    bbox=bbox,
                    method=distance_method
                )
                z_est = float(dist_info.get("z", 0.0))
                d3d_est = float(dist_info.get("distance", 0.0))
            except Exception:
                bh = max(1.0, float(bbox[3] - bbox[1]))
                z_est = float(dist_estimator.fy * 0.70 / bh)
                d3d_est = z_est

            all_predictions.append({
                "image_stem": stem,
                "class_name": cname,
                "confidence": conf,
                "bbox": bbox,
                "z": z_est,
                "distance_3d": d3d_est
            })

        if verbose or (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f"Processed [{idx + 1}/{len(image_paths)}] images...")

    eval_duration = time.perf_counter() - start_eval_time
    print(f"\nInference complete in {eval_duration:.2f}s ({len(image_paths) / max(0.001, eval_duration):.1f} FPS)\n")

    # Evaluate metrics per-class
    per_class_results: Dict[str, Dict[str, Any]] = {}
    pooled_distance_errors_z: List[float] = []
    pooled_distance_rel_errors_z: List[float] = []
    pooled_distance_sq_errors_z: List[float] = []
    pooled_distance_errors_3d: List[float] = []
    pooled_distance_rel_errors_3d: List[float] = []
    pooled_distance_sq_errors_3d: List[float] = []

    total_tp_all = 0
    total_fp_all = 0
    total_gt_all = 0
    ap_list = []

    for cname in class_names:
        # Collect ground truths for this class
        gts_for_class: Dict[str, List[Dict[str, Any]]] = {}
        n_gt_class = 0
        for stem, gt_list in gts_by_image.items():
            filtered = [g for g in gt_list if g["class_name"] == cname]
            gts_for_class[stem] = filtered
            n_gt_class += len(filtered)

        # Collect predictions for this class
        preds_for_class = [p for p in all_predictions if p["class_name"] == cname]
        # Sort predictions descending by confidence score
        preds_for_class.sort(key=lambda x: x["confidence"], reverse=True)

        n_pred_class = len(preds_for_class)
        matched_gt_per_img: Dict[str, Set[int]] = {stem: set() for stem in gts_by_image.keys()}

        tp_array = np.zeros(n_pred_class, dtype=np.float32)
        fp_array = np.zeros(n_pred_class, dtype=np.float32)

        class_dist_err_z: List[float] = []
        class_dist_rel_z: List[float] = []
        class_dist_sq_z: List[float] = []
        class_dist_err_3d: List[float] = []
        class_dist_rel_3d: List[float] = []
        class_dist_sq_3d: List[float] = []

        for p_idx, pred in enumerate(preds_for_class):
            stem = pred["image_stem"]
            gt_candidates = gts_for_class.get(stem, [])

            if not gt_candidates:
                fp_array[p_idx] = 1.0
                continue

            # Compute IoU with each candidate GT
            best_iou = -1.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_candidates):
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh:
                if best_gt_idx not in matched_gt_per_img[stem]:
                    # True Positive match
                    tp_array[p_idx] = 1.0
                    matched_gt_per_img[stem].add(best_gt_idx)

                    # Compute distance error metrics against matched ground-truth
                    matched_gt = gt_candidates[best_gt_idx]
                    err_z = abs(pred["z"] - matched_gt["Z"])
                    rel_z = err_z / max(0.01, matched_gt["Z"])
                    sq_z = (pred["z"] - matched_gt["Z"]) ** 2

                    class_dist_err_z.append(err_z)
                    class_dist_rel_z.append(rel_z)
                    class_dist_sq_z.append(sq_z)

                    err_3d = abs(pred["distance_3d"] - matched_gt["distance_3d"])
                    rel_3d = err_3d / max(0.01, matched_gt["distance_3d"])
                    sq_3d = (pred["distance_3d"] - matched_gt["distance_3d"]) ** 2

                    class_dist_err_3d.append(err_3d)
                    class_dist_rel_3d.append(rel_3d)
                    class_dist_sq_3d.append(sq_3d)
                else:
                    # Duplicate detection for an already matched GT
                    fp_array[p_idx] = 1.0
            else:
                fp_array[p_idx] = 1.0

        tp_count = int(np.sum(tp_array))
        fp_count = int(np.sum(fp_array))
        total_tp_all += tp_count
        total_fp_all += fp_count
        total_gt_all += n_gt_class

        # Precision, Recall, and AP50
        if n_pred_class > 0:
            precision_val = float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
        else:
            precision_val = 0.0

        if n_gt_class > 0:
            recall_val = float(tp_count / n_gt_class)
            cum_tp = np.cumsum(tp_array)
            cum_fp = np.cumsum(fp_array)
            precisions = cum_tp / (cum_tp + cum_fp)
            recalls = cum_tp / n_gt_class
            ap50 = compute_ap(recalls, precisions)
            ap_list.append(ap50)
        else:
            recall_val = 0.0
            ap50 = 0.0

        # Distance Metrics for class
        if class_dist_err_z:
            mae_z = float(np.mean(class_dist_err_z))
            mre_z = float(np.mean(class_dist_rel_z)) * 100.0  # percentage
            rmse_z = float(np.sqrt(np.mean(class_dist_sq_z)))
            pooled_distance_errors_z.extend(class_dist_err_z)
            pooled_distance_rel_errors_z.extend(class_dist_rel_z)
            pooled_distance_sq_errors_z.extend(class_dist_sq_z)
        else:
            mae_z, mre_z, rmse_z = 0.0, 0.0, 0.0

        if class_dist_err_3d:
            mae_3d = float(np.mean(class_dist_err_3d))
            mre_3d = float(np.mean(class_dist_rel_3d)) * 100.0
            rmse_3d = float(np.sqrt(np.mean(class_dist_sq_3d)))
            pooled_distance_errors_3d.extend(class_dist_err_3d)
            pooled_distance_rel_errors_3d.extend(class_dist_rel_3d)
            pooled_distance_sq_errors_3d.extend(class_dist_sq_3d)
        else:
            mae_3d, mre_3d, rmse_3d = 0.0, 0.0, 0.0

        per_class_results[cname] = {
            "gt_count": n_gt_class,
            "pred_count": n_pred_class,
            "tp": tp_count,
            "fp": fp_count,
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "ap50": round(ap50, 4),
            "mAP50": round(ap50, 4),
            "map50": round(ap50, 4),
            "distance_mae_z": round(mae_z, 3),
            "distance_mre_z_pct": round(mre_z, 2),
            "distance_rmse_z": round(rmse_z, 3),
            "distance_mae_3d": round(mae_3d, 3),
            "distance_mre_3d_pct": round(mre_3d, 2),
            "distance_rmse_3d": round(rmse_3d, 3),
            "distance_mae": round(mae_z, 3),
            "distance_mre": round(mre_z, 2),
            "distance_rmse": round(rmse_z, 3),
            "mae": round(mae_z, 3),
            "mre": round(mre_z, 2),
            "rmse": round(rmse_z, 3)
        }

    # Overall Summary Metrics
    overall_map50 = float(np.mean(ap_list)) if ap_list else 0.0
    overall_precision = float(total_tp_all / (total_tp_all + total_fp_all)) if (total_tp_all + total_fp_all) > 0 else 0.0
    overall_recall = float(total_tp_all / total_gt_all) if total_gt_all > 0 else 0.0

    overall_mae_z = float(np.mean(pooled_distance_errors_z)) if pooled_distance_errors_z else 0.0
    overall_mre_z = float(np.mean(pooled_distance_rel_errors_z)) * 100.0 if pooled_distance_rel_errors_z else 0.0
    overall_rmse_z = float(np.sqrt(np.mean(pooled_distance_sq_errors_z))) if pooled_distance_sq_errors_z else 0.0

    overall_mae_3d = float(np.mean(pooled_distance_errors_3d)) if pooled_distance_errors_3d else 0.0
    overall_mre_3d = float(np.mean(pooled_distance_rel_errors_3d)) * 100.0 if pooled_distance_rel_errors_3d else 0.0
    overall_rmse_3d = float(np.sqrt(np.mean(pooled_distance_sq_errors_3d))) if pooled_distance_sq_errors_3d else 0.0

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "model_path": str(model_path),
        "num_val_images": len(image_paths),
        "iou_threshold": iou_thresh,
        "confidence_threshold": conf_thresh,
        "distance_method": distance_method,
        "overall": {
            "gt_count": total_gt_all,
            "pred_count": len(all_predictions),
            "tp": total_tp_all,
            "fp": total_fp_all,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "mAP50": round(overall_map50, 4),
            "map50": round(overall_map50, 4),
            "mAP": round(overall_map50, 4),
            "distance_mae_z": round(overall_mae_z, 3),
            "distance_mre_z_pct": round(overall_mre_z, 2),
            "distance_rmse_z": round(overall_rmse_z, 3),
            "distance_mae_3d": round(overall_mae_3d, 3),
            "distance_mre_3d_pct": round(overall_mre_3d, 2),
            "distance_rmse_3d": round(overall_rmse_3d, 3),
            "distance_mae": round(overall_mae_z, 3),
            "distance_mre": round(overall_mre_z, 2),
            "distance_rmse": round(overall_rmse_z, 3),
            "mae": round(overall_mae_z, 3),
            "mre": round(overall_mre_z, 2),
            "rmse": round(overall_rmse_z, 3)
        },
        "per_class": per_class_results
    }

    # Print Formatted Evaluation Table
    header = (
        f"{'Class':<12} | {'GT':<4} | {'TP':<4} | {'FP':<4} | "
        f"{'Prec':<7} | {'Recall':<7} | {'mAP50':<7} | "
        f"{'MAE (Z)':<8} | {'MRE (Z)':<8} | {'RMSE (Z)':<8}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for cname, res in per_class_results.items():
        print(
            f"{cname:<12} | {res['gt_count']:<4} | {res['tp']:<4} | {res['fp']:<4} | "
            f"{res['precision']:<7.4f} | {res['recall']:<7.4f} | {res['ap50']:<7.4f} | "
            f"{res['distance_mae_z']:<6.2f}m | {res['distance_mre_z_pct']:<7.2f}% | {res['distance_rmse_z']:<6.2f}m"
        )

    print(separator)
    ov = summary["overall"]
    print(
        f"{'OVERALL':<12} | {ov['gt_count']:<4} | {ov['tp']:<4} | {ov['fp']:<4} | "
        f"{ov['precision']:<7.4f} | {ov['recall']:<7.4f} | {ov['mAP50']:<7.4f} | "
        f"{ov['distance_mae_z']:<6.2f}m | {ov['distance_mre_z_pct']:<7.2f}% | {ov['distance_rmse_z']:<6.2f}m"
    )
    print(separator + "\n")

    print("Key Perception Metrics Summary:")
    print(f"  * Detection mAP@50:                {ov['mAP50'] * 100:.2f}%")
    print(f"  * Detection Precision:             {ov['precision'] * 100:.2f}%")
    print(f"  * Detection Recall:                {ov['recall'] * 100:.2f}%")
    print(f"  * Distance Estimation MAE (Z):     {ov['distance_mae_z']:.3f} m")
    print(f"  * Distance Estimation MRE (Z):     {ov['distance_mre_z_pct']:.2f}%")
    print(f"  * Distance Estimation RMSE (Z):    {ov['distance_rmse_z']:.3f} m")
    print(f"  * Distance Estimation MAE (3D):    {ov['distance_mae_3d']:.3f} m")
    print(f"  * Distance Estimation MRE (3D):    {ov['distance_mre_3d_pct']:.2f}%")
    print(f"  * Distance Estimation RMSE (3D):   {ov['distance_rmse_3d']:.3f} m\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Automated CLI Evaluation for Detection mAP50 and Distance Estimation Metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_dataset/dataset.yaml",
        help="Path to dataset.yaml configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8_navigation.pt",
        help="Path to model (.pt or .onnx)"
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Optional override for ground truth JSON directory"
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default="configs/camera_params.yaml",
        help="Path to camera intrinsic/extrinsic YAML"
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.50,
        help="IoU matching threshold (default: 0.50)"
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="Detection NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cpu', or None for auto)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["hybrid", "pinhole", "ground"],
        help="Distance estimation method"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output evaluation results JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-image logs"
    )
    args = parser.parse_args()

    results = evaluate_dataset(
        dataset_yaml=args.dataset,
        model_path=args.model,
        gt_dir_override=args.gt_dir,
        camera_config=args.camera_config,
        iou_thresh=args.iou_thresh,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        device=args.device,
        distance_method=args.method,
        verbose=args.verbose
    )

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[evaluate_distance] Saved full metrics JSON to {out_p}")


if __name__ == "__main__":
    main()
