"""
Unit tests for Automated Detection & Metric Distance Estimation Evaluation Tooling.
"""

import math
import numpy as np
import pytest
from pathlib import Path

from benchmarks.evaluate_distance import (
    box_iou,
    compute_ap,
    normalize_class_name,
    resolve_dataset_paths,
    evaluate_dataset
)


def test_box_iou_identical():
    box1 = [10.0, 20.0, 110.0, 120.0]
    box2 = [10.0, 20.0, 110.0, 120.0]
    iou = box_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-5) == 1.0


def test_box_iou_disjoint():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert box_iou(box1, box2) == 0.0


def test_box_iou_partial():
    # Box1: [0, 0, 10, 10], area = 100
    # Box2: [5, 0, 15, 10], area = 100
    # Intersection: [5, 0, 10, 10], area = 5 * 10 = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 ~ 0.333333
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    iou = box_iou(box1, box2)
    assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0


def test_compute_ap_perfect():
    recalls = np.array([0.5, 1.0])
    precisions = np.array([1.0, 1.0])
    ap = compute_ap(recalls, precisions)
    assert pytest.approx(ap, rel=1e-5) == 1.0


def test_compute_ap_decreasing():
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    ap = compute_ap(recalls, precisions)
    assert 0.0 < ap < 1.0


def test_compute_ap_empty():
    assert compute_ap(np.array([]), np.array([])) == 0.0


def test_normalize_class_name():
    assert normalize_class_name("Stop Sign") == "stop_sign"
    assert normalize_class_name("cone ") == "cone"
    assert normalize_class_name("Traffic-Cone") == "traffic_cone"


def test_resolve_dataset_paths():
    root_dir, val_images_dir, gt_dir, class_names = resolve_dataset_paths(
        "data/synthetic_dataset/dataset.yaml"
    )
    assert root_dir.exists()
    assert val_images_dir.exists()
    assert gt_dir.exists()
    assert "cone" in class_names
    assert "barrier" in class_names
    assert "stop_sign" in class_names


def test_evaluate_dataset_end_to_end(tmp_path):
    out_json = str(tmp_path / "eval_results.json")
    results = evaluate_dataset(
        dataset_yaml="data/synthetic_dataset/dataset.yaml",
        model_path="models/yolov8_navigation.pt",
        camera_config="configs/camera_params.yaml",
        iou_thresh=0.50,
        conf_thresh=0.25,
        device="cpu"
    )

    assert "overall" in results
    assert "per_class" in results

    ov = results["overall"]
    assert "mAP50" in ov
    assert "precision" in ov
    assert "recall" in ov
    assert "distance_mae_z" in ov
    assert "distance_mre_z_pct" in ov
    assert "distance_rmse_z" in ov

    # Verify metric sanity
    assert 0.0 <= ov["precision"] <= 1.0
    assert 0.0 <= ov["recall"] <= 1.0
    assert 0.0 <= ov["mAP50"] <= 1.0
    assert ov["distance_mae_z"] >= 0.0
    assert ov["distance_rmse_z"] >= 0.0

    # Verify classes
    for c in ["cone", "barrier", "stop_sign"]:
        assert c in results["per_class"]
        c_res = results["per_class"][c]
        assert c_res["gt_count"] > 0
        assert c_res["distance_mae_z"] >= 0.0

    # Verify standard aliases
    assert "map50" in ov
    assert "distance_mae" in ov
    assert "mae" in ov
    assert "rmse" in ov


def test_box_iou_degenerate():
    # Zero area box
    assert box_iou([10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]) == 0.0
    # Inverted / negative coordinates
    assert box_iou([20.0, 20.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]) == 0.0


def test_compute_ap_zero_recall():
    recalls = np.array([0.0, 0.0])
    precisions = np.array([0.0, 0.0])
    assert compute_ap(recalls, precisions) == 0.0


def test_compute_ap_single_point():
    recalls = np.array([1.0])
    precisions = np.array([1.0])
    assert pytest.approx(compute_ap(recalls, precisions), rel=1e-5) == 1.0


def test_evaluate_dataset_alternative_methods():
    for method in ["pinhole", "ground"]:
        results = evaluate_dataset(
            dataset_yaml="data/synthetic_dataset/dataset.yaml",
            model_path="models/yolov8_navigation.pt",
            camera_config="configs/camera_params.yaml",
            iou_thresh=0.50,
            conf_thresh=0.30,
            device="cpu",
            distance_method=method
        )
        assert "overall" in results
        ov = results["overall"]
        assert ov["distance_mae_z"] >= 0.0
        assert ov["distance_rmse_z"] >= 0.0
