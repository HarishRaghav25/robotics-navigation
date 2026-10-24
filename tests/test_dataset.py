"""
Unit tests for Dataset Pipeline, BDD100K octagonal stop sign filtering,
and YAML portability.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import yaml
import pytest

from data.dataset import BDD100KParser, SyntheticRoboticsGenerator


def test_dataset_yaml_relative_path():
    """Verify data/synthetic_dataset/dataset.yaml has clean relative path: ."""
    yaml_path = Path("data/synthetic_dataset/dataset.yaml")
    assert yaml_path.exists(), "dataset.yaml does not exist"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("path") == ".", f"Expected path to be '.', got {cfg.get('path')}"
    assert "train" in cfg
    assert "val" in cfg
    assert "names" in cfg


def test_build_dataset_yaml_export(tmp_path):
    """Verify build_dataset exports path: . into dataset.yaml."""
    gen = SyntheticRoboticsGenerator()
    out_dir = tmp_path / "test_synth_dataset"
    gen.build_dataset(str(out_dir), train_count=2, val_count=1)

    generated_yaml = out_dir / "dataset.yaml"
    assert generated_yaml.exists()

    with open(generated_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("path") == ".", f"build_dataset should write path: '.', got {cfg.get('path')}"


def test_build_dataset_zero_count(tmp_path):
    """Verify build_dataset gracefully handles 0 counts without crashing."""
    gen = SyntheticRoboticsGenerator()
    out_dir = tmp_path / "empty_dataset"
    gen.build_dataset(str(out_dir), train_count=0, val_count=0)
    assert (out_dir / "dataset.yaml").exists()


def test_bdd100k_stop_sign_metadata_filtering():
    """Verify BDD100KParser differentiates octagonal stop signs from general traffic signs via metadata."""
    # 1. Explicit stop sign category variations
    assert BDD100KParser.is_octagonal_stop_sign({"category": "stop sign", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}) is True
    assert BDD100KParser.is_octagonal_stop_sign({"category": "stop_sign", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}) is True
    assert BDD100KParser.is_octagonal_stop_sign({"category": "stopsign", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}) is True

    # 2. General traffic sign with signType == stop
    label_ts_stop = {
        "category": "traffic sign",
        "attributes": {"trafficSignType": "stop"},
        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_ts_stop) is True

    # 3. General traffic sign with uppercase / case-insensitive shape == OCTAGONAL
    label_ts_octagon = {
        "category": "traffic sign",
        "attributes": {"shape": "OCTAGONAL"},
        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_ts_octagon) is True

    # 4. General traffic sign without stop attributes (e.g. speed limit / street name)
    label_ts_speed = {
        "category": "traffic sign",
        "attributes": {"trafficSignType": "speed limit", "shape": "rectangle"},
        "box2d": {"x1": 100, "y1": 100, "x2": 150, "y2": 220}
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_ts_speed) is False

    # 5. Non-traffic sign category
    label_car = {"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}
    assert BDD100KParser.is_octagonal_stop_sign(label_car) is False

    # 6. Robustness against None attributes
    label_null_attrs = {
        "category": "traffic sign",
        "attributes": None,
        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_null_attrs) is False

    # 7. Robustness against None box2d
    assert BDD100KParser.is_octagonal_stop_sign({"category": "traffic sign", "box2d": None}) is False

    # 8. Robustness against non-dict label
    assert BDD100KParser.is_octagonal_stop_sign(None) is False
    assert BDD100KParser.is_octagonal_stop_sign("invalid") is False

    # 9. Poly2d octagonal vertices
    label_poly = {
        "category": "traffic sign",
        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "poly2d": [{"vertices": [[0, 0]] * 8}]
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_poly) is True

    # 10. Malformed poly2d with None vertices
    label_poly_null = {
        "category": "traffic sign",
        "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "poly2d": [{"vertices": None}, None]
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_poly_null) is False


def test_bdd100k_stop_sign_visual_verification():
    """Verify BDD100KParser checks visual color and contour when inspecting crops."""
    # Create an image with an octagonal red stop sign crop
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw red octagon in center
    center = (100, 100)
    radius = 40
    angles = np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)[:-1]
    pts = [[int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))] for a in angles]
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], (25, 25, 210))  # Red in BGR

    label_stop = {
        "category": "traffic sign",
        "box2d": {"x1": 60, "y1": 60, "x2": 140, "y2": 140}
    }
    assert BDD100KParser.is_octagonal_stop_sign(label_stop, image=img) is True

    # Create an image with a blue/green rectangle (guide sign)
    img_guide = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img_guide, (60, 60), (140, 140), (200, 100, 10), -1)  # Blue-ish BGR
    assert BDD100KParser.is_octagonal_stop_sign(label_stop, image=img_guide) is False

    # Grayscale image (single-channel) should not raise an OpenCV exception
    img_gray = np.zeros((200, 200), dtype=np.uint8)
    assert BDD100KParser.is_octagonal_stop_sign(label_stop, image=img_gray) is False


def test_bdd100k_convert_to_yolo(tmp_path):
    """Verify convert_to_yolo converts cones, barriers, octagonal stop signs, and ignores generic traffic signs."""
    ann_file = tmp_path / "bdd_labels.json"
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "yolo_output"

    labels_data = [
        {
            "name": "sample1.jpg",
            "labels": [
                {"category": "cone", "box2d": {"x1": 100, "y1": 100, "x2": 150, "y2": 200}},
                {"category": "barrier", "box2d": {"x1": 200, "y1": 200, "x2": 350, "y2": 300}},
                {
                    "category": "traffic sign",
                    "attributes": {"trafficSignType": "stop"},
                    "box2d": {"x1": 400, "y1": 100, "x2": 450, "y2": 150}
                },
                {"category": "stop_sign", "box2d": {"x1": 150, "y1": 50, "x2": 250, "y2": 150}},
                {
                    "category": "traffic sign",
                    "attributes": {"trafficSignType": "speed limit"},
                    "box2d": {"x1": 500, "y1": 100, "x2": 550, "y2": 200}
                },
                {"category": "pedestrian", "box2d": {"x1": 600, "y1": 100, "x2": 650, "y2": 300}},
                # Malformed entries that must be handled safely
                {"category": "cone", "box2d": None},
                {"category": "traffic sign", "attributes": None, "box2d": None},
                None,
                # Inverted coordinates that should be ordered correctly
                {"category": "cone", "box2d": {"x1": 350, "y1": 200, "x2": 300, "y2": 100}}
            ]
        }
    ]

    with open(ann_file, "w") as f:
        json.dump(labels_data, f)

    parser = BDD100KParser(str(ann_file), str(img_dir))
    count = parser.convert_to_yolo(str(out_dir))
    assert count == 1

    label_txt = out_dir / "labels" / "sample1.txt"
    assert label_txt.exists()

    with open(label_txt, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Should have 5 valid objects: cone (0), barrier (1), stop sign (2), stop_sign (2), inverted cone (0).
    # Speed limit, pedestrian, and malformed items should NOT be present.
    assert len(lines) == 5
    class_ids = [int(line.split()[0]) for line in lines]
    assert 0 in class_ids  # cone
    assert 1 in class_ids  # barrier
    assert 2 in class_ids  # stop_sign

    # Check that all coordinates are strictly non-negative and normalized (0.0 to 1.0)
    for line in lines:
        parts = [float(p) for p in line.split()]
        cid, cx, cy, bw, bh = parts
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < bw <= 1.0
        assert 0.0 < bh <= 1.0
