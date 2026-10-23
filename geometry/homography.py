"""
Homography & Perspective Transform for Bird's-Eye View (BEV) Robotics Mapping.
Warps camera road perspective into an orthographic top-down Euclidean grid.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional


class BirdEyeViewTransformer:
    """
    Computes perspective transform homography to warp forward-facing camera
    views into metric top-down Bird's-Eye View (BEV) grids for obstacle avoidance.
    """

    def __init__(
        self,
        img_width: int = 1280,
        img_height: int = 720,
        bev_width: int = 400,
        bev_height: int = 400,
        metric_width: float = 6.0,    # -3.0m to +3.0m lateral span
        metric_length: float = 20.0    # 0.0m to 20.0m longitudinal forward span
    ):
        self.img_w = img_width
        self.img_h = img_height
        self.bev_w = bev_width
        self.bev_h = bev_height
        self.metric_w = metric_width
        self.metric_l = metric_length

        # Source trapezoid points on camera ground plane (pixels)
        # Near road (y near bottom) and Far road (y near horizon)
        y_near = int(img_height * 0.95)
        y_far = int(img_height * 0.55)

        src_pts = np.float32([
            [img_width * 0.15, y_near],      # Bottom-left
            [img_width * 0.85, y_near],      # Bottom-right
            [img_width * 0.58, y_far],       # Top-right
            [img_width * 0.42, y_far]        # Top-left
        ])

        # Destination top-down rectangle (BEV pixels)
        dst_pts = np.float32([
            [bev_width * 0.20, bev_height * 0.95],
            [bev_width * 0.80, bev_height * 0.95],
            [bev_width * 0.80, bev_height * 0.05],
            [bev_width * 0.20, bev_height * 0.05]
        ])

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def warp_image(self, img: np.ndarray) -> np.ndarray:
        """Warp camera image into bird's-eye view."""
        return cv2.warpPerspective(img, self.H, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def metric_to_bev_pixel(self, x_metric: float, z_metric: float) -> Tuple[int, int]:
        """Convert real metric coordinates (X right, Z forward in meters) to BEV grid pixels."""
        # Normalize lateral X [-metric_w/2, +metric_w/2] -> [0, bev_w]
        u = int(self.bev_w / 2.0 + (x_metric / (self.metric_w / 2.0)) * (self.bev_w * 0.45))
        # Normalize longitudinal Z [0, metric_l] -> [bev_h, 0] (0m is bottom, 20m is top)
        v = int(self.bev_h * 0.95 - (z_metric / self.metric_l) * (self.bev_h * 0.90))

        u = max(0, min(self.bev_w - 1, u))
        v = max(0, min(self.bev_h - 1, v))
        return u, v

    def render_radar_grid(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Render a clean top-down metric radar map with distance rings,
        robot position, and detected obstacles.
        """
        radar = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
        radar[:] = (20, 24, 28)  # Dark slate background

        center_u = self.bev_w // 2
        bottom_v = int(self.bev_h * 0.95)

        # Draw distance range rings: 2m, 5m, 10m, 15m
        for dist_m in [2.0, 5.0, 10.0, 15.0]:
            _, v_ring = self.metric_to_bev_pixel(0.0, dist_m)
            radius = bottom_v - v_ring
            if radius > 0:
                # Ring color: Red for 2m, Yellow for 5m, Green for >5m
                color = (0, 0, 180) if dist_m <= 2.0 else ((0, 200, 220) if dist_m <= 5.0 else (80, 120, 80))
                cv2.circle(radar, (center_u, bottom_v), radius, color, 1, cv2.LINE_AA)
                cv2.putText(radar, f"{int(dist_m)}m", (center_u + radius - 25, bottom_v - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

        # Draw lateral guide lines
        cv2.line(radar, (center_u, bottom_v), (center_u, 0), (50, 60, 70), 1, cv2.LINE_AA)
        cv2.line(radar, (0, bottom_v), (self.bev_w, bottom_v), (50, 60, 70), 1, cv2.LINE_AA)

        # Draw Robot footprint (small triangle or rectangle)
        robot_pts = np.array([
            [center_u, bottom_v - 12],
            [center_u - 8, bottom_v + 4],
            [center_u + 8, bottom_v + 4]
        ], np.int32)
        cv2.fillPoly(radar, [robot_pts], (0, 255, 120))  # Green robot icon

        # Color palette for classes
        class_colors = {
            "cone": (20, 120, 245),       # Orange
            "barrier": (220, 200, 0),     # Cyan / yellow
            "stop_sign": (40, 40, 230)     # Red
        }

        # Plot detected objects
        for det in detections:
            x_m = det.get("x", 0.0)
            z_m = det.get("z", det.get("distance", 5.0))
            cls = det.get("class_name", "cone").lower().replace(" ", "_")

            u_obj, v_obj = self.metric_to_bev_pixel(x_m, z_m)
            col = class_colors.get(cls, (200, 200, 200))

            # Draw obstacle marker
            cv2.circle(radar, (u_obj, v_obj), 6, col, -1, cv2.LINE_AA)
            cv2.circle(radar, (u_obj, v_obj), 8, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw distance label
            lbl = f"{z_m:.1f}m"
            cv2.putText(radar, lbl, (u_obj + 8, v_obj + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Radar Title & border
        cv2.rectangle(radar, (0, 0), (self.bev_w - 1, self.bev_h - 1), (100, 110, 120), 1)
        cv2.putText(radar, "BEV METRIC RADAR", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        return radar
