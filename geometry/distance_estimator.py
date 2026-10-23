"""
Geometry-based Distance Estimation for Robotics Navigation.
Implements:
1. Pinhole Camera Model (object height and width priors)
2. Ground Plane Contact Geometry (Flat-Earth Inverse Perspective Model)
3. Sensor Fusion / Hybrid Multi-Hypothesis Estimator
"""

import math
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List


class DistanceEstimator:
    """
    Estimates 3D metric distance to detected navigation objects (cones, barriers, stop signs)
    using calibrated camera intrinsics, mounting extrinsics, and canonical object priors.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Default camera intrinsics (1280x720 HD)
        self.fx: float = 960.0
        self.fy: float = 960.0
        self.cx: float = 640.0
        self.cy: float = 360.0
        self.image_width: int = 1280
        self.image_height: int = 720

        # Mounting extrinsics
        self.camera_height: float = 1.0   # meters above ground
        self.camera_pitch: float = 0.0    # radians (downward tilt is positive)

        # Canonical object physical dimensions in meters
        self.object_specs: Dict[str, Dict[str, float]] = {
            "cone": {"height": 0.70, "width": 0.38, "min_dist": 0.3, "max_dist": 50.0},
            "barrier": {"height": 0.85, "width": 1.60, "min_dist": 0.5, "max_dist": 60.0},
            "stop_sign": {"height": 0.75, "width": 0.75, "min_dist": 0.5, "max_dist": 70.0}
        }

        if config_path and Path(config_path).exists():
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cam = cfg.get("camera", {})
        self.fx = float(cam.get("fx", self.fx))
        self.fy = float(cam.get("fy", self.fy))
        self.cx = float(cam.get("cx", self.cx))
        self.cy = float(cam.get("cy", self.cy))
        self.image_width = int(cam.get("image_width", self.image_width))
        self.image_height = int(cam.get("image_height", self.image_height))
        self.camera_height = float(cam.get("camera_height", self.camera_height))
        self.camera_pitch = float(cam.get("camera_pitch", self.camera_pitch))

        objs = cfg.get("object_dimensions", {})
        for name, specs in objs.items():
            self.object_specs[name] = specs

    def estimate_pinhole_height(self, class_name: str, bbox_height: float) -> float:
        """
        Estimate distance Z based on vertical pinhole projection:
        Z = (fy * RealHeight) / BboxHeight
        """
        if bbox_height <= 1.0:
            return 99.0
        spec = self.object_specs.get(class_name.lower(), {"height": 0.70})
        real_h = spec.get("height", 0.70)
        z = (self.fy * real_h) / bbox_height
        return z

    def estimate_pinhole_width(self, class_name: str, bbox_width: float) -> float:
        """
        Estimate distance Z based on horizontal pinhole projection:
        Z = (fx * RealWidth) / BboxWidth
        """
        if bbox_width <= 1.0:
            return 99.0
        spec = self.object_specs.get(class_name.lower(), {"width": 0.50})
        real_w = spec.get("width", 0.50)
        z = (self.fx * real_w) / bbox_width
        return z

    def estimate_ground_plane(self, y_bottom: float) -> float:
        """
        Estimate longitudinal distance Z via Flat-Earth ground contact ray geometry:
        alpha = arctan((y_bottom - cy) / fy)
        total_pitch = camera_pitch + alpha
        Z = camera_height / tan(total_pitch)
        """
        # Pixel offset below principal point
        dy = y_bottom - self.cy
        if dy <= 1.0:
            # At or above horizon
            return 80.0

        alpha = math.atan2(dy, self.fy)
        total_pitch = self.camera_pitch + alpha

        if total_pitch <= 0.01:
            return 80.0

        z = self.camera_height / math.tan(total_pitch)
        return max(0.2, z)

    def estimate_distance(
        self,
        class_name: str,
        bbox: List[float],
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Compute robust 3D metric distance to detected object.

        Args:
            class_name: "cone", "barrier", or "stop_sign"
            bbox: [x1, y1, x2, y2] in pixel coordinates
            method: "pinhole", "ground", or "hybrid"

        Returns:
            Dictionary with:
            - distance: estimated 3D Euclidean distance (meters)
            - z: longitudinal distance (meters)
            - x: lateral offset from camera axis (meters)
            - formatted_label: e.g. "Cone, 1.5m" or "Stop Sign, 3.2m"
        """
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        u_center = (x1 + x2) / 2.0
        norm_name = class_name.lower().replace(" ", "_")

        # 1. Pinhole height distance
        z_pinhole_h = self.estimate_pinhole_height(norm_name, bh)

        # 2. Ground contact distance
        z_ground = self.estimate_ground_plane(float(y2))

        # 3. Hybrid fusion
        if method == "pinhole":
            z_final = z_pinhole_h
        elif method == "ground":
            z_final = z_ground
        else:
            # Weighted multi-hypothesis fusion
            if norm_name == "stop_sign":
                # Stop signs are elevated on poles; height prior is reliable
                w_pinhole = 0.85
                w_ground = 0.15
            elif norm_name == "barrier":
                # Barriers sit on ground; ground contact & height are both valid
                w_pinhole = 0.40
                w_ground = 0.60
            else:  # cone
                # Traffic cones rest directly on the ground
                w_pinhole = 0.45
                w_ground = 0.55

            z_final = w_pinhole * z_pinhole_h + w_ground * z_ground

        # Clamp to realistic min/max ranges
        spec = self.object_specs.get(norm_name, {"min_dist": 0.3, "max_dist": 60.0})
        min_d = spec.get("min_dist", 0.3)
        max_d = spec.get("max_dist", 60.0)
        z_final = max(min_d, min(max_d, z_final))

        # Lateral 3D offset (X)
        x_lateral = ((u_center - self.cx) * z_final) / self.fx

        # True 3D Euclidean distance from camera optical center
        dist_3d = math.sqrt(x_lateral**2 + z_final**2 + (self.camera_height)**2)

        # Format label strictly matching requirement:
        # e.g. "Cone, 1.5m" or "Stop Sign, 3.2m"
        display_name = {
            "cone": "Cone",
            "barrier": "Barrier",
            "stop_sign": "Stop Sign",
            "stopsign": "Stop Sign"
        }.get(norm_name, class_name.title())

        formatted_label = f"{display_name}, {z_final:.1f}m"

        return {
            "distance": round(dist_3d, 2),
            "z": round(z_final, 2),
            "x": round(x_lateral, 2),
            "z_pinhole": round(z_pinhole_h, 2),
            "z_ground": round(z_ground, 2),
            "formatted_label": formatted_label
        }
