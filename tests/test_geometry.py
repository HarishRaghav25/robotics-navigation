"""
Unit tests for geometric distance estimation, epipolar geometry,
BEV homography, and optical flow tracking.
"""

import math
import numpy as np
import pytest
from geometry.distance_estimator import DistanceEstimator
from geometry.epipolar import StereoDepthEstimator
from geometry.homography import BirdEyeViewTransformer
from geometry.optical_flow import ObstacleTracker


class TestGeometry:

    def setup_method(self):
        self.estimator = DistanceEstimator()

    def test_pinhole_height_calculation(self):
        # Cone real height = 0.70m, fy = 960.0, if Z = 2.0m, bh should be (960 * 0.70) / 2.0 = 336.0 px
        bh = 336.0
        z = self.estimator.estimate_pinhole_height("cone", bh)
        assert pytest.approx(z, rel=1e-2) == 2.0

    def test_pinhole_stop_sign(self):
        # Stop sign real height = 0.75m, fy = 960.0, Z = 5.0m -> bh = (960 * 0.75) / 5.0 = 144.0 px
        bh = 144.0
        z = self.estimator.estimate_pinhole_height("stop_sign", bh)
        assert pytest.approx(z, rel=1e-2) == 5.0

    def test_ground_plane_estimation(self):
        # Camera height = 1.0m, cy = 360.0, fy = 960.0
        # If object contact is at y = 552.0 -> dy = 192.0 -> alpha = atan(192/960) = atan(0.2)
        # Z = 1.0 / tan(atan(0.2)) = 1.0 / 0.2 = 5.0m
        y_bottom = 552.0
        z = self.estimator.estimate_ground_plane(y_bottom)
        assert pytest.approx(z, rel=1e-2) == 5.0

    def test_formatted_label_output(self):
        # Test required string label format: "Cone, 1.5m", "Stop Sign, 3.2m", "Barrier, 2.8m"
        res_cone = self.estimator.estimate_distance("cone", [100, 100, 200, 300])
        assert "Cone, " in res_cone["formatted_label"]
        assert res_cone["formatted_label"].endswith("m")

        res_stop = self.estimator.estimate_distance("stop_sign", [100, 100, 200, 300])
        assert "Stop Sign, " in res_stop["formatted_label"]

        res_barrier = self.estimator.estimate_distance("barrier", [100, 100, 200, 300])
        assert "Barrier, " in res_barrier["formatted_label"]

    def test_epipolar_disparity_depth(self):
        stereo = StereoDepthEstimator(focal_length=960.0, baseline=0.20)
        # Z = (f * B) / d -> for d = 38.4 px -> Z = (960 * 0.20) / 38.4 = 5.0m
        disp_map = np.full((100, 100), 38.4, dtype=np.float32)
        depth_map = stereo.disparity_to_depth(disp_map)
        assert pytest.approx(depth_map[50, 50], rel=1e-2) == 5.0

    def test_bev_homography_warp(self):
        bev = BirdEyeViewTransformer(1280, 720, 400, 400)
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        warped = bev.warp_image(dummy_img)
        assert warped.shape == (400, 400, 3)

        # Test radar grid generation
        dets = [{"class_name": "cone", "x": 0.5, "z": 4.5}]
        radar = bev.render_radar_grid(dets)
        assert radar.shape == (400, 400, 3)

    def test_optical_flow_tracking_and_ttc(self):
        tracker = ObstacleTracker(fps=30.0)
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Frame 1: cone at Z = 6.0m
        dets1 = [{"class_name": "cone", "bbox": [500, 300, 600, 450], "z": 6.0}]
        up1 = tracker.update(frame1, dets1)
        assert len(up1) == 1
        assert up1[0]["track_id"] == 1

        # Frame 2: cone moved closer to Z = 5.0m in 1/30s (velocity ~ 30 m/s -> TTC ~ 5.0 / 30 ~ 0.17s)
        dets2 = [{"class_name": "cone", "bbox": [500, 300, 600, 460], "z": 5.0}]
        up2 = tracker.update(frame2, dets2)
        assert len(up2) == 1
        assert up2[0]["track_id"] == 1
        assert up2[0]["vz"] > 0
        assert up2[0]["ttc"] < 5.0
        assert up2[0]["hazard_level"] in ["CRITICAL", "CAUTION"]
