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

    def test_bottom_truncated_obstacle_distance(self):
        # A traffic cone at 1.5m where camera height = 1.0m
        # Ground base is at v = 360 + 960 * 1.0 / 1.5 = 1000 px (clipped to 719!)
        # Real width = 0.38m -> bw = 960 * 0.38 / 1.5 = 243.2 px
        # Top is at v = 360 + 960 * 0.3 / 1.5 = 552 px
        # Bbox: x1=518, y1=552, x2=761, y2=719
        bbox = [518, 552, 761, 719]
        res = self.estimator.estimate_distance("cone", bbox)
        assert res["bottom_clipped"] is True
        # Must be accurate to ~1.5m (not 3.3m as the old buggy version estimated)
        assert pytest.approx(res["z"], abs=0.1) == 1.5
        assert res["formatted_label"] == "Cone, 1.5m"

    def test_stop_sign_adaptive_priors(self):
        # 1. Stop sign including pole (aspect > 1.6) at 3.2m:
        # bh = 960 * 2.05 / 3.2 = 615 px, bw = 960 * 0.75 / 3.2 = 225 px
        bbox_pole = [500, 50, 725, 665]
        res_pole = self.estimator.estimate_distance("stop_sign", bbox_pole)
        assert pytest.approx(res_pole["z"], abs=0.1) == 3.2
        assert res_pole["formatted_label"] == "Stop Sign, 3.2m"

        # 2. Stop sign face only (aspect ~ 1.0) elevated in air at 3.2m:
        # bh = 225 px, bw = 225 px, y2 in air
        bbox_face = [500, 100, 725, 325]
        res_face = self.estimator.estimate_distance("stop_sign", bbox_face)
        assert pytest.approx(res_face["z"], abs=0.1) == 3.2
        assert res_face["formatted_label"] == "Stop Sign, 3.2m"

    def test_optical_flow_real_pattern_tracking(self):
        tracker = ObstacleTracker(fps=30.0)
        # Create textured frames with known motion shift
        np.random.seed(42)
        texture = np.random.randint(50, 220, (100, 100), dtype=np.uint8)

        frame1 = np.zeros((720, 1280), dtype=np.uint8)
        frame1[200:300, 400:500] = texture

        # Frame 2: shift texture by dx = +10, dy = +6
        frame2 = np.zeros((720, 1280), dtype=np.uint8)
        frame2[206:306, 410:510] = texture

        dets1 = [{"class_name": "cone", "bbox": [400, 200, 500, 300], "z": 5.0}]
        tracker.update(frame1, dets1)

        dets2 = [{"class_name": "cone", "bbox": [410, 206, 510, 306], "z": 4.8}]
        up2 = tracker.update(frame2, dets2)

        assert len(up2) == 1
        flow_vec = up2[0]["flow_vector"]
        # Optical flow vector should closely match dx=10, dy=6
        assert pytest.approx(flow_vec[0], abs=2.0) == 10.0
        assert pytest.approx(flow_vec[1], abs=2.0) == 6.0
        assert up2[0]["optical_flow_points"] > 0
