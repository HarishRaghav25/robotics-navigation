"""
Dataset management, BDD100K parser, and Synthetic Robotics Navigation Scenario Generator.
Generates realistic perspective-correct scenes with known 3D ground truth for training,
testing, and geometric distance validation.
"""

import os
import json
import math
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

CLASS_NAMES = ["cone", "barrier", "stop_sign"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Canonical physical dimensions (in meters: height, width)
OBJECT_SPECS = {
    "cone": {"height": 0.70, "width": 0.38, "color": (30, 105, 237)},         # Orange (BGR)
    "barrier": {"height": 0.85, "width": 1.60, "color": (220, 220, 220)},     # Light grey / chevrons
    "stop_sign": {"height": 0.75, "width": 0.75, "color": (36, 36, 192)}       # Red (BGR)
}


class BDD100KParser:
    """Parser to extract navigation relevant classes from BDD100K detection labels."""

    BDD_CATEGORY_MAP = {
        "stop sign": "stop_sign",
        "stop_sign": "stop_sign",
        "stopsign": "stop_sign",
        "stop-sign": "stop_sign",
        "barrier": "barrier",
        "cone": "cone",
        "traffic cone": "cone",
        "traffic_cone": "cone",
        "traffic-cone": "cone"
    }

    def __init__(self, annotation_file: str, images_dir: str):
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)

    @staticmethod
    def is_octagonal_stop_sign(label: Dict, image: Optional[np.ndarray] = None) -> bool:
        """
        Determine whether a label represents an octagonal stop sign rather than
        a general traffic sign. Evaluates label category, metadata attributes,
        polygon geometry (if present), aspect ratio, and visual cues (color/shape
        contour analysis on the image crop).
        """
        if not isinstance(label, dict):
            return False

        category = str(label.get("category", "")).lower().strip()
        norm_cat = category.replace("-", " ").replace("_", " ")

        # Explicit stop sign category
        if norm_cat in ("stop sign", "stopsign") or category in ("stop sign", "stop_sign", "stopsign", "stop-sign"):
            return True

        # Check explicit attributes / subcategory if present
        attrs = label.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}

        sign_type = str(attrs.get("trafficSignType") or attrs.get("signType") or attrs.get("type") or "").lower()
        shape = str(attrs.get("shape") or attrs.get("trafficSignShape") or "").lower()
        name = str(label.get("name") or "").lower()
        subcat = str(label.get("subCategory") or label.get("subcategory") or "").lower()

        if "stop" in sign_type or "stop" in name or "stop" in subcat:
            return True
        if shape in ("octagonal", "octagon") or "octagon" in shape:
            return True

        # If not labeled as general traffic sign, it's not a stop sign
        if norm_cat not in ("traffic sign", "trafficsign"):
            return False

        # If box2d is missing or invalid, cannot inspect geometry or image
        b = label.get("box2d")
        if not isinstance(b, dict):
            return False

        try:
            x1 = float(b["x1"])
            y1 = float(b["y1"])
            x2 = float(b["x2"])
            y2 = float(b["y2"])
        except (KeyError, ValueError, TypeError):
            return False

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        bw = x2 - x1
        bh = y2 - y1

        # Check aspect ratio for octagonal face: stop signs are roughly 1:1
        if bh <= 0 or bw <= 0:
            return False
        aspect_ratio = bw / bh
        if not (0.70 <= aspect_ratio <= 1.35):
            return False

        # Check polygon approximation if poly2d annotation exists
        poly2d = label.get("poly2d")
        if isinstance(poly2d, list):
            for poly in poly2d:
                if isinstance(poly, dict):
                    vertices = poly.get("vertices")
                    if isinstance(vertices, (list, tuple)) and 7 <= len(vertices) <= 9:
                        return True
                elif isinstance(poly, (list, tuple)) and 7 <= len(poly) <= 9:
                    return True

        # If image crop is available, perform visual color & shape verification
        if image is not None and isinstance(image, np.ndarray) and image.size > 0:
            # Handle grayscale conversion if single-channel
            if image.ndim == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] >= 3:
                img_bgr = image[:, :, :3]
            else:
                img_bgr = None

            if img_bgr is not None:
                h_img, w_img = img_bgr.shape[:2]
                ix1 = max(0, min(w_img - 1, int(round(x1))))
                ix2 = max(0, min(w_img, int(round(x2))))
                iy1 = max(0, min(h_img - 1, int(round(y1))))
                iy2 = max(0, min(h_img, int(round(y2))))

                if (ix2 - ix1) >= 8 and (iy2 - iy1) >= 8:
                    crop = img_bgr[iy1:iy2, ix1:ix2]
                    if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                        # Red color range in HSV (split across hue 0/180)
                        mask1 = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([12, 255, 255]))
                        mask2 = cv2.inRange(hsv, np.array([160, 40, 30]), np.array([180, 255, 255]))
                        red_mask = mask1 | mask2

                        total_pixels = crop.shape[0] * crop.shape[1]
                        red_pixels = cv2.countNonZero(red_mask)
                        red_ratio = red_pixels / float(total_pixels) if total_pixels > 0 else 0.0

                        # Stop signs have a prominent red field (typically > 20% of bounding box)
                        if red_ratio >= 0.20:
                            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_cnt = max(contours, key=cv2.contourArea)
                                cnt_area = cv2.contourArea(largest_cnt)
                                if cnt_area >= 0.25 * total_pixels:
                                    peri = cv2.arcLength(largest_cnt, True)
                                    approx = cv2.approxPolyDP(largest_cnt, 0.025 * peri, True)
                                    # Regular octagon under perspective/discretization yields 6 to 10 vertices
                                    if 6 <= len(approx) <= 10 or red_ratio >= 0.35:
                                        return True

        return False

    def convert_to_yolo(self, output_dir: str, max_samples: Optional[int] = None) -> int:
        """Convert BDD100K annotations to YOLO format."""
        if not self.annotation_file.exists():
            print(f"[BDD100KParser] Annotation file not found: {self.annotation_file}")
            return 0

        out_path = Path(output_dir)
        labels_dir = out_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            if max_samples and count >= max_samples:
                break

            img_name = item.get("name", "")
            labels = item.get("labels", [])
            if not isinstance(labels, list):
                continue
            yolo_lines = []

            # Attempt to load image if available for visual verification
            img_path = self.images_dir / img_name
            img = cv2.imread(str(img_path)) if (self.images_dir.exists() and img_path.exists()) else None

            for label in labels:
                if not isinstance(label, dict):
                    continue
                category = str(label.get("category", "")).lower().strip()
                norm_cat = category.replace("-", " ").replace("_", " ")

                # Filter specifically for octagonal stop signs rather than mapping all general traffic signs
                if norm_cat in ("traffic sign", "trafficsign"):
                    if self.is_octagonal_stop_sign(label, image=img):
                        target_cls = "stop_sign"
                    else:
                        continue
                else:
                    target_cls = self.BDD_CATEGORY_MAP.get(category) or self.BDD_CATEGORY_MAP.get(norm_cat)

                if not target_cls or target_cls not in CLASS_TO_ID:
                    continue

                b = label.get("box2d")
                if not isinstance(b, dict):
                    continue

                try:
                    x1 = float(b["x1"])
                    y1 = float(b["y1"])
                    x2 = float(b["x2"])
                    y2 = float(b["y2"])
                except (KeyError, ValueError, TypeError):
                    continue

                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Assuming 1280x720 if not specified
                w_img, h_img = (float(img.shape[1]), float(img.shape[0])) if img is not None else (1280.0, 720.0)
                if w_img <= 0.0 or h_img <= 0.0:
                    continue

                # Clip coordinates to image boundary
                x1 = max(0.0, min(w_img, x1))
                x2 = max(0.0, min(w_img, x2))
                y1 = max(0.0, min(h_img, y1))
                y2 = max(0.0, min(h_img, y2))

                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                if bw <= 0.0 or bh <= 0.0:
                    continue

                cx = ((x1 + x2) / 2.0) / w_img
                cy = ((y1 + y2) / 2.0) / h_img

                cls_id = CLASS_TO_ID[target_cls]
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if yolo_lines:
                stem = Path(img_name).stem
                with open(labels_dir / f"{stem}.txt", "w") as f_out:
                    f_out.write("\n".join(yolo_lines))
                count += 1

        print(f"[BDD100KParser] Converted {count} files to YOLO format.")
        return count


class SyntheticRoboticsGenerator:
    """
    Renders realistic synthetic robotics navigation scenes using real projective geometry.
    Objects are placed at exact 3D coordinates (X, Y, Z) in the camera frame,
    providing ground truth distance and 2D bounding boxes for verification.
    """

    def __init__(self, camera_config_path: Optional[str] = None):
        if camera_config_path and os.path.exists(camera_config_path):
            with open(camera_config_path, "r") as f:
                cfg = yaml.safe_load(f)["camera"]
            self.fx = cfg.get("fx", 960.0)
            self.fy = cfg.get("fy", 960.0)
            self.cx = cfg.get("cx", 640.0)
            self.cy = cfg.get("cy", 360.0)
            self.width = cfg.get("image_width", 1280)
            self.height = cfg.get("image_height", 720)
            self.cam_height = cfg.get("camera_height", 1.0)
        else:
            self.fx = 960.0
            self.fy = 960.0
            self.cx = 640.0
            self.cy = 360.0
            self.width = 1280
            self.height = 720
            self.cam_height = 1.0

    def project_3d_to_2d(self, X: float, Y: float, Z: float) -> Tuple[int, int]:
        """Project 3D camera coordinate (meters) to 2D image pixel."""
        if Z <= 0.05:
            Z = 0.05
        u = int(self.fx * (X / Z) + self.cx)
        v = int(self.fy * (Y / Z) + self.cy)
        return u, v

    def draw_road_scene(self) -> np.ndarray:
        """Render a realistic outdoor road / ground navigation background."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky gradient (light blue to horizon haze)
        horizon_y = int(self.cy)
        for y in range(horizon_y):
            ratio = y / horizon_y
            b = int(240 - 50 * ratio)
            g = int(210 - 40 * ratio)
            r = int(180 - 40 * ratio)
            img[y, :] = [b, g, r]

        # Ground / Asphalt gradient (horizon to bottom)
        for y in range(horizon_y, self.height):
            ratio = (y - horizon_y) / (self.height - horizon_y)
            gray = int(70 + 35 * ratio)
            # Add slight texture noise
            img[y, :] = [gray, gray + 2, gray + 4]

        # Draw road boundaries / lane markers in perspective
        pts_road = np.array([
            [int(self.cx - 80), horizon_y],
            [int(self.cx + 80), horizon_y],
            [self.width - 80, self.height],
            [80, self.height]
        ], np.int32)
        cv2.fillPoly(img, [pts_road], (55, 58, 60))

        # Lane markings (dashed center line)
        for step in range(1, 10):
            z_near = 2.0 + step * 2.5
            z_far = z_near + 1.2
            u1, v1 = self.project_3d_to_2d(0.0, self.cam_height, z_near)
            u2, v2 = self.project_3d_to_2d(0.0, self.cam_height, z_far)
            if 0 <= v1 < self.height and 0 <= v2 < self.height:
                cv2.line(img, (u1, v1), (u2, v2), (220, 220, 220), thickness=max(2, int(8 / (z_near * 0.4))))

        # Side road lines
        for side in [-1.8, 1.8]:
            pts_line = []
            for z_val in np.linspace(2.0, 35.0, 15):
                u, v = self.project_3d_to_2d(side, self.cam_height, z_val)
                pts_line.append([u, v])
            cv2.polylines(img, [np.array(pts_line, np.int32)], False, (245, 245, 245), thickness=2)

        return img

    def draw_traffic_cone(self, img: np.ndarray, X: float, Z: float) -> Tuple[int, int, int, int]:
        """Render a 3D perspective traffic cone at ground coordinate (X, Z)."""
        real_h = OBJECT_SPECS["cone"]["height"]
        real_w = OBJECT_SPECS["cone"]["width"]
        Y_base = self.cam_height
        Y_top = self.cam_height - real_h

        # Base center and top center
        u_base, v_base = self.project_3d_to_2d(X, Y_base, Z)
        u_top, v_top = self.project_3d_to_2d(X, Y_top, Z)

        h_px = max(6, v_base - v_top)
        w_px = max(4, int(self.fx * (real_w / Z)))

        x1 = max(0, u_base - w_px // 2)
        x2 = min(self.width - 1, u_base + w_px // 2)
        y1 = max(0, v_top)
        y2 = min(self.height - 1, v_base)

        if x2 <= x1 or y2 <= y1:
            return x1, y1, x2, y2

        # Cone shadow
        cv2.ellipse(img, (u_base, v_base), (w_px // 2 + 2, max(2, w_px // 6)), 0, 0, 360, (30, 30, 30), -1)

        # Base square plate
        base_h = max(2, int(h_px * 0.08))
        cv2.rectangle(img, (x1, y2 - base_h), (x2, y2), (20, 20, 20), -1)

        # Conical body (Orange)
        cone_pts = np.array([
            [u_top - max(1, w_px // 8), y1],
            [u_top + max(1, w_px // 8), y1],
            [x2 - max(1, w_px // 10), y2 - base_h],
            [x1 + max(1, w_px // 10), y2 - base_h]
        ], np.int32)
        cv2.fillPoly(img, [cone_pts], (20, 100, 245))

        # White reflective stripes
        for band_ratio_y, band_h_ratio in [(0.35, 0.14), (0.60, 0.12)]:
            band_y1 = int(y1 + h_px * band_ratio_y)
            band_y2 = int(band_y1 + h_px * band_h_ratio)
            # interpolate width at band heights
            w1 = int(w_px * (band_ratio_y * 0.8 + 0.2))
            w2 = int(w_px * ((band_ratio_y + band_h_ratio) * 0.8 + 0.2))
            band_pts = np.array([
                [u_base - w1 // 2, band_y1],
                [u_base + w1 // 2, band_y1],
                [u_base + w2 // 2, band_y2],
                [u_base - w2 // 2, band_y2]
            ], np.int32)
            cv2.fillPoly(img, [band_pts], (240, 240, 245))

        return x1, y1, x2, y2

    def draw_stop_sign(self, img: np.ndarray, X: float, Z: float) -> Tuple[int, int, int, int]:
        """Render a 3D perspective Stop Sign with pole at coordinate (X, Z)."""
        real_h = OBJECT_SPECS["stop_sign"]["height"]
        real_w = OBJECT_SPECS["stop_sign"]["width"]
        pole_h = 1.3  # Pole height from ground

        Y_ground = self.cam_height
        Y_sign_bottom = self.cam_height - pole_h
        Y_sign_top = Y_sign_bottom - real_h

        u_base, v_ground = self.project_3d_to_2d(X, Y_ground, Z)
        u_center, v_sign_center = self.project_3d_to_2d(X, (Y_sign_bottom + Y_sign_top) / 2.0, Z)

        h_px = max(8, int(self.fy * (real_h / Z)))
        w_px = max(8, int(self.fx * (real_w / Z)))
        radius = max(4, h_px // 2)

        # Bounding box includes sign and pole
        x1 = max(0, u_center - radius)
        x2 = min(self.width - 1, u_center + radius)
        y1 = max(0, v_sign_center - radius)
        y2 = min(self.height - 1, v_ground)

        if x2 <= x1 or y2 <= y1:
            return x1, y1, x2, y2

        # Draw pole (metallic grey)
        pole_w = max(2, int(w_px * 0.08))
        cv2.rectangle(img, (u_center - pole_w // 2, v_sign_center), (u_center + pole_w // 2, v_ground), (160, 160, 165), -1)

        # Stop Sign regular octagon points
        angles = np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)[:-1]
        pts_octagon = []
        for a in angles:
            ox = int(u_center + radius * np.cos(a))
            oy = int(v_sign_center + radius * np.sin(a))
            pts_octagon.append([ox, oy])
        pts_octagon = np.array(pts_octagon, np.int32)

        # Red outer octagon
        cv2.fillPoly(img, [pts_octagon], (25, 25, 210))
        # White inner border
        cv2.polylines(img, [pts_octagon], True, (250, 250, 250), thickness=max(1, int(radius * 0.08)))

        # "STOP" text if sign is sufficiently large
        if radius >= 14:
            font_scale = radius / 38.0
            thick = max(1, int(font_scale * 2.2))
            text_size = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_DUPLEX, font_scale, thick)[0]
            tx = u_center - text_size[0] // 2
            ty = v_sign_center + text_size[1] // 2
            cv2.putText(img, "STOP", (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

        return x1, y1, x2, y2

    def draw_barrier(self, img: np.ndarray, X: float, Z: float) -> Tuple[int, int, int, int]:
        """Render a 3D perspective construction barrier with diagonal hazard stripes."""
        real_h = OBJECT_SPECS["barrier"]["height"]
        real_w = OBJECT_SPECS["barrier"]["width"]

        Y_base = self.cam_height
        Y_top = self.cam_height - real_h

        u_center, v_base = self.project_3d_to_2d(X, Y_base, Z)
        _, v_top = self.project_3d_to_2d(X, Y_top, Z)

        h_px = max(8, v_base - v_top)
        w_px = max(12, int(self.fx * (real_w / Z)))

        x1 = max(0, u_center - w_px // 2)
        x2 = min(self.width - 1, u_center + w_px // 2)
        y1 = max(0, v_top)
        y2 = min(self.height - 1, v_base)

        if x2 <= x1 or y2 <= y1:
            return x1, y1, x2, y2

        # Draw legs
        leg_w = max(2, int(w_px * 0.06))
        cv2.rectangle(img, (x1 + leg_w, y1 + int(h_px * 0.7)), (x1 + 2 * leg_w, y2), (40, 40, 40), -1)
        cv2.rectangle(img, (x2 - 2 * leg_w, y1 + int(h_px * 0.7)), (x2 - leg_w, y2), (40, 40, 40), -1)

        # Barrier main panel
        panel_y2 = y1 + int(h_px * 0.75)
        panel = img[y1:panel_y2, x1:x2].copy()
        if panel.size > 0:
            # White background
            panel[:] = (245, 245, 245)
            # Orange diagonal stripes
            stripe_w = max(4, int(w_px * 0.12))
            for sx in range(-h_px, w_px + h_px, stripe_w * 2):
                pts = np.array([
                    [sx, 0],
                    [sx + stripe_w, 0],
                    [sx + stripe_w + h_px, panel_y2 - y1],
                    [sx + h_px, panel_y2 - y1]
                ], np.int32)
                cv2.fillPoly(panel, [pts], (20, 110, 240))
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (30, 30, 30), 1)
            img[y1:panel_y2, x1:x2] = panel

        return x1, y1, x2, y2

    def generate_scene(self, scene_id: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate a complete scene with road background and 2 to 6 obstacles.
        Returns the rendered image and list of object ground truth metadata:
        [{class_id, class_name, bbox: [x1, y1, x2, y2], distance_3d, distance_z, X, Y, Z}]
        """
        img = self.draw_road_scene()
        objects = []

        # Decide objects count
        num_objects = random.randint(2, 5)
        # Sample positions sorted by distance Z descending so distant objects are drawn first
        z_positions = sorted([random.uniform(2.5, 26.0) for _ in range(num_objects)], reverse=True)

        for z in z_positions:
            cls_name = random.choice(CLASS_NAMES)
            # Lateral position X depends on class
            if cls_name == "stop_sign":
                # Typically on road shoulder (right or left)
                x = random.choice([random.uniform(1.8, 3.2), random.uniform(-3.2, -1.8)])
            elif cls_name == "barrier":
                x = random.choice([random.uniform(-1.8, -0.6), random.uniform(0.6, 2.0)])
            else:  # cone
                x = random.uniform(-1.6, 1.6)

            # Draw object based on class
            if cls_name == "cone":
                box = self.draw_traffic_cone(img, x, z)
            elif cls_name == "stop_sign":
                box = self.draw_stop_sign(img, x, z)
            else:
                box = self.draw_barrier(img, x, z)

            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1

            # Only record valid bounding boxes
            if w_box >= 4 and h_box >= 6:
                dist_3d = math.sqrt(x * x + z * z + (self.cam_height)**2)
                objects.append({
                    "class_id": CLASS_TO_ID[cls_name],
                    "class_name": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "distance_3d": round(dist_3d, 2),
                    "distance_z": round(z, 2),
                    "X": round(x, 2),
                    "Y": round(self.cam_height, 2),
                    "Z": round(z, 2)
                })

        return img, objects

    def build_dataset(self, output_root: str, train_count: int = 120, val_count: int = 30):
        """Build and save full YOLO dataset structure."""
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        for split, count in [("train", train_count), ("val", val_count)]:
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            gt_dir = root / split / "ground_truth"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Dataset] Generating {count} synthetic scenes for {split}...")
            for i in range(count):
                img, objects = self.generate_scene(scene_id=i)
                base_name = f"synth_{split}_{i:04d}"

                # Save image
                cv2.imwrite(str(img_dir / f"{base_name}.jpg"), img)

                # Save YOLO labels
                yolo_lines = []
                for obj in objects:
                    x1, y1, x2, y2 = obj["bbox"]
                    cx = ((x1 + x2) / 2.0) / self.width
                    cy = ((y1 + y2) / 2.0) / self.height
                    w = (x2 - x1) / self.width
                    h = (y2 - y1) / self.height
                    yolo_lines.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                with open(lbl_dir / f"{base_name}.txt", "w") as f_lbl:
                    f_lbl.write("\n".join(yolo_lines))

                # Save ground truth metadata for distance evaluation
                with open(gt_dir / f"{base_name}.json", "w") as f_gt:
                    json.dump(objects, f_gt, indent=2)

        # Write dataset.yaml for YOLO
        yaml_content = {
            "path": ".",
            "train": "train/images",
            "val": "val/images",
            "names": {i: name for i, name in enumerate(CLASS_NAMES)}
        }
        with open(root / "dataset.yaml", "w") as f_yaml:
            yaml.dump(yaml_content, f_yaml, sort_keys=False)

        print(f"[Dataset] Dataset successfully generated at {root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic robotics dataset")
    parser.add_argument("--output", type=str, default="data/synthetic_dataset", help="Output directory")
    parser.add_argument("--train", type=int, default=100, help="Train samples")
    parser.add_argument("--val", type=int, default=25, help="Val samples")
    args = parser.parse_args()

    gen = SyntheticRoboticsGenerator("configs/camera_params.yaml")
    gen.build_dataset(args.output, train_count=args.train, val_count=args.val)
