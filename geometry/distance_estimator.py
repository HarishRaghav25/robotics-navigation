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
import numpy as np


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
            "stop_sign": {"height": 0.75, "width": 0.75, "min_dist": 0.5, "max_dist": 70.0},
            "stopsign": {"height": 0.75, "width": 0.75, "min_dist": 0.5, "max_dist": 70.0}
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
            norm = name.lower().replace(" ", "_")
            self.object_specs[norm] = specs
            if norm == "stop_sign":
                self.object_specs["stopsign"] = specs

    def estimate_pinhole_height(self, class_name: str, bbox_height: float) -> float:
        """
        Estimate distance Z based on vertical pinhole projection:
        Z = (fy * RealHeight) / BboxHeight
        """
        if not isinstance(bbox_height, (int, float, np.floating, np.integer)) or isinstance(bbox_height, bool) or not math.isfinite(float(bbox_height)) or float(bbox_height) <= 1.0:
            return 99.0
        norm_name = class_name.lower().replace(" ", "_")
        if norm_name == "stopsign":
            norm_name = "stop_sign"
        spec = self.object_specs.get(norm_name, {"height": 0.70})
        real_h = spec.get("height", 0.70)
        z = (self.fy * real_h) / float(bbox_height)
        return z

    def estimate_pinhole_width(self, class_name: str, bbox_width: float) -> float:
        """
        Estimate distance Z based on horizontal pinhole projection:
        Z = (fx * RealWidth) / BboxWidth
        """
        if not isinstance(bbox_width, (int, float, np.floating, np.integer)) or isinstance(bbox_width, bool) or not math.isfinite(float(bbox_width)) or float(bbox_width) <= 1.0:
            return 99.0
        norm_name = class_name.lower().replace(" ", "_")
        if norm_name == "stopsign":
            norm_name = "stop_sign"
        spec = self.object_specs.get(norm_name, {"width": 0.50})
        real_w = spec.get("width", 0.50)
        z = (self.fx * real_w) / float(bbox_width)
        return z

    def estimate_ground_plane(self, y_bottom: float) -> float:
        """
        Estimate longitudinal distance Z via Flat-Earth ground contact ray geometry:
        alpha = arctan((y_bottom - cy) / fy)
        total_pitch = camera_pitch + alpha
        Z = camera_height / tan(total_pitch)
        """
        if not isinstance(y_bottom, (int, float, np.floating, np.integer)) or isinstance(y_bottom, bool) or not math.isfinite(float(y_bottom)):
            return 80.0

        # Pixel offset below principal point
        dy = float(y_bottom) - self.cy
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
        Fuses horizontal pinhole (width), vertical pinhole (height), and ground-plane ray geometry.
        Automatically handles boundary clipping (near-field obstacles < 2.5m) and elevated stop signs.

        Args:
            class_name: "cone", "barrier", or "stop_sign"
            bbox: [x1, y1, x2, y2] in pixel coordinates
            method: "pinhole", "ground", or "hybrid"

        Returns:
            Dictionary with distance, z, x, and formatted_label e.g. "Cone, 1.5m", "Stop Sign, 3.2m"
        """
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Bounding box must contain exactly 4 coordinates [x1, y1, x2, y2], got {bbox}")

        try:
            x1, y1, x2, y2 = bbox
            if any(isinstance(v, bool) for v in [x1, y1, x2, y2]):
                raise TypeError("Booleans are not valid coordinates")
            coords = [float(v) for v in [x1, y1, x2, y2]]
            if not all(math.isfinite(c) for c in coords):
                raise ValueError("Non-finite coordinates")
        except (TypeError, ValueError, IndexError) as err:
            raise ValueError(f"Bounding box coordinates must be finite numbers, got {bbox}") from err

        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Bounding box coordinates must be non-degenerate (x2 > x1, y2 > y1), got x1={x1}, x2={x2}, y1={y1}, y2={y2}")

        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        u_center = (x1 + x2) / 2.0
        norm_name = class_name.lower().replace(" ", "_")
        if norm_name == "stopsign":
            norm_name = "stop_sign"

        # Boundary truncation flags
        bottom_clipped = y2 >= (self.image_height - 3)
        top_clipped = y1 <= 3
        left_clipped = x1 <= 3
        right_clipped = x2 >= (self.image_width - 3)
        horiz_clipped = left_clipped or right_clipped

        # Ground FOV limit at bottom of image
        dy_bot = max(1.0, float(self.image_height - 1 - self.cy))
        alpha_bot = math.atan2(dy_bot, self.fy)
        z_bottom_fov = self.camera_height / math.tan(self.camera_pitch + alpha_bot)

        # 1. Horizontal pinhole distance (width)
        z_pinhole_w = self.estimate_pinhole_width(norm_name, bw)

        # 2. Vertical pinhole distance (height) with adaptive prior
        aspect = bh / bw
        if norm_name == "stop_sign":
            # If aspect ratio > 1.6, detection encompasses the support post (total ~2.05m)
            # If aspect ratio <= 1.6, detection is only the octagonal face (0.75m)
            real_h = 2.05 if aspect > 1.6 else 0.75
            z_pinhole_h = (self.fy * real_h) / bh
        else:
            z_pinhole_h = self.estimate_pinhole_height(norm_name, bh)

        # 3. Ground contact distance
        z_ground = self.estimate_ground_plane(float(y2))
        ground_valid = (y2 - self.cy) > 1.0 and z_ground < 79.0

        # 4. Multi-Cue Hybrid Estimation
        if method == "pinhole":
            if not horiz_clipped and (bottom_clipped or top_clipped):
                z_final = z_pinhole_w
            else:
                z_final = 0.5 * z_pinhole_h + 0.5 * z_pinhole_w
        elif method == "ground":
            z_final = z_ground
        else:
            # Intelligent Multi-Cue Hybrid Fusion
            if bottom_clipped:
                # Near-field obstacle where base is clipped by bottom frame edge (< 2.6m)
                # Ground contact and box height are truncated and overestimate distance.
                if not horiz_clipped:
                    z_final = z_pinhole_w
                else:
                    z_final = min(z_pinhole_w, z_bottom_fov)
                z_final = min(z_final, z_bottom_fov)

            elif top_clipped and not bottom_clipped:
                # Top clipped, but ground contact and width are valid
                if horiz_clipped:
                    z_final = z_ground if ground_valid else z_pinhole_w
                elif norm_name != "stop_sign":
                    z_final = 0.60 * z_ground + 0.40 * z_pinhole_w if ground_valid else z_pinhole_w
                else:
                    z_final = z_pinhole_w

            elif horiz_clipped:
                # Object is clipped by left or right frame edge (width is truncated)
                # Pinhole width will overestimate distance; rely on height and ground contact
                if norm_name == "stop_sign":
                    if aspect > 1.6 and ground_valid:
                        z_final = 0.65 * z_pinhole_h + 0.35 * z_ground
                    else:
                        z_final = z_pinhole_h
                elif norm_name == "barrier":
                    z_final = (0.50 * z_pinhole_h + 0.50 * z_ground) if ground_valid else z_pinhole_h
                else:  # cone
                    z_final = (0.45 * z_pinhole_h + 0.55 * z_ground) if ground_valid else z_pinhole_h

            elif norm_name == "stop_sign":
                # Fully visible stop sign
                if aspect > 1.6 and ground_valid:
                    # Sign + post: ground contact and height are both valid
                    z_final = 0.45 * z_pinhole_w + 0.35 * z_pinhole_h + 0.20 * z_ground
                else:
                    # Octagon only or in air: ground contact is invalid!
                    z_final = 0.50 * z_pinhole_w + 0.50 * z_pinhole_h

            elif norm_name == "barrier":
                # Jersey / pedestrian barrier sitting on ground
                if ground_valid:
                    z_final = 0.30 * z_pinhole_w + 0.35 * z_pinhole_h + 0.35 * z_ground
                else:
                    z_final = 0.50 * z_pinhole_w + 0.50 * z_pinhole_h

            else:  # cone
                # Traffic cone resting on ground
                if ground_valid:
                    z_final = 0.35 * z_pinhole_w + 0.30 * z_pinhole_h + 0.35 * z_ground
                else:
                    z_final = 0.50 * z_pinhole_w + 0.50 * z_pinhole_h

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
            "z_pinhole_w": round(z_pinhole_w, 2),
            "z_ground": round(z_ground, 2),
            "formatted_label": formatted_label,
            "bottom_clipped": bottom_clipped
        }
