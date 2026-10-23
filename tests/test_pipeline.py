"""
Unit tests for End-to-End Perception Pipeline and Annotator.
"""

import numpy as np
import pytest
from pipeline.annotator import NavigationAnnotator


def test_annotator_label_format():
    annotator = NavigationAnnotator(draw_bev_inset=False, draw_ground_rays=False)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = [
        {"class_name": "cone", "bbox": [100, 200, 150, 300], "z": 1.5, "formatted_label": "Cone, 1.5m"},
        {"class_name": "stop_sign", "bbox": [300, 100, 380, 250], "z": 3.2, "formatted_label": "Stop Sign, 3.2m"},
        {"class_name": "barrier", "bbox": [500, 250, 650, 380], "z": 2.8, "formatted_label": "Barrier, 2.8m"}
    ]

    annotated = annotator.annotate(frame, detections, fps=35.0)
    assert annotated.shape == (720, 1280, 3)
    # Check that pixels are not all black anymore
    assert np.count_nonzero(annotated) > 0


def test_risk_color_thresholds():
    annotator = NavigationAnnotator()
    c_crit = annotator.get_risk_color(1.2)
    c_caut = annotator.get_risk_color(3.5)
    c_safe = annotator.get_risk_color(8.0)

    assert c_crit == annotator.color_critical
    assert c_caut == annotator.color_caution
    assert c_safe == annotator.color_safe
