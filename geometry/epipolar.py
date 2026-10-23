"""
Epipolar Geometry & Stereo Disparity-Depth Estimation.
Derivation:
Given a calibrated, rectified stereo camera pair with baseline B (meters)
and focal length f (pixels):
For a 3D scene point P(X, Y, Z), its projections in the left and right cameras are:
  u_L = f * (X / Z) + c_x
  u_R = f * ((X - B) / Z) + c_x

The disparity d is defined as the horizontal pixel difference:
  d = u_L - u_R = f * (B / Z)

Rearranging gives the fundamental disparity-to-depth relationship:
  Z = (f * B) / d

Lateral and vertical coordinates:
  X = (u_L - c_x) * Z / f
  Y = (v_L - c_y) * Z / f
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


class StereoDepthEstimator:
    """
    Computes disparity maps and metric depth using rectified stereo pairs
    via Epipolar Geometry and Semi-Global Block Matching (StereoSGBM).
    """

    def __init__(
        self,
        focal_length: float = 960.0,
        baseline: float = 0.20,
        cx: float = 640.0,
        cy: float = 360.0,
        num_disparities: int = 64,
        block_size: int = 5
    ):
        self.f = focal_length
        self.b = baseline
        self.cx = cx
        self.cy = cy
        self.num_disparities = num_disparities
        self.block_size = block_size

        # Initialize OpenCV StereoSGBM Matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """Compute disparity map from left and right rectified images."""
        if len(img_left.shape) == 3:
            gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_l = img_left
            gray_r = img_right

        disp_16s = self.stereo.compute(gray_l, gray_r)
        # SGBM returns 16-bit fixed point with 4 fractional bits (divided by 16)
        disp = disp_16s.astype(np.float32) / 16.0
        return disp

    def disparity_to_depth(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to metric depth map Z = (f * B) / d.
        Invalid/negative disparities are set to 0.0 (masked).
        """
        depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        valid = disparity_map > 0.5
        depth_map[valid] = (self.f * self.b) / disparity_map[valid]
        return depth_map

    def depth_to_pointcloud(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Project 2D depth map to 3D metric point cloud (X, Y, Z).
        Returns array of shape (N, 3).
        """
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        valid = (depth_map > 0.2) & (depth_map < 50.0)
        z = depth_map[valid]
        x = (u[valid] - self.cx) * z / self.f
        y = (v[valid] - self.cy) * z / self.f

        points_3d = np.column_stack([x, y, z])
        return points_3d

    def get_roi_depth(self, depth_map: np.ndarray, bbox: list) -> float:
        """Extract robust median metric depth within bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        roi = depth_map[y1:y2, x1:x2]
        valid_pixels = roi[roi > 0.2]
        if len(valid_pixels) == 0:
            return 0.0
        return float(np.median(valid_pixels))
