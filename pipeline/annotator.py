"""
Annotation and Visual HUD Rendering for Robotics Navigation.
Renders bounding boxes with strict required formatting:
  "Cone, 1.5m"
  "Stop Sign, 3.2m"
  "Barrier, 2.8m"
Includes distance-based risk color coding, HUD telemetry, and BEV Radar inset.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class NavigationAnnotator:
    """
    Renders navigation annotations, distance tags, collision warnings,
    and optional Bird's-Eye View radar overlays on camera frames.
    """

    def __init__(self, draw_bev_inset: bool = True, draw_ground_rays: bool = True):
        self.draw_bev_inset = draw_bev_inset
        self.draw_ground_rays = draw_ground_rays

        # Color palette (BGR)
        self.color_critical = (30, 30, 230)     # Red (< 2.0m)
        self.color_caution = (20, 210, 255)     # Amber (2.0m - 5.0m)
        self.color_safe = (50, 205, 50)         # Green (> 5.0m)

    def get_risk_color(self, distance: float) -> Tuple[int, int, int]:
        """Determine bounding box color based on distance threshold."""
        if distance < 2.0:
            return self.color_critical
        elif distance < 5.0:
            return self.color_caution
        else:
            return self.color_safe

    def draw_corner_box(
        self,
        img: np.ndarray,
        bbox: List[int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 15
    ):
        """Draw aesthetic bounding box with highlighted corner brackets."""
        x1, y1, x2, y2 = bbox
        # Main box with semi-transparent or subtle outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Highlight corners
        cl = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
        th = thickness

        # Top-left
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, th, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, th, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, th, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, th, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, th, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, th, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, th, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, th, cv2.LINE_AA)

    def draw_text_badge(
        self,
        img: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.55,
        thickness: int = 1
    ):
        """Render readable text badge with solid background pill."""
        x, y = origin
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 4

        # Pill background
        bx1 = x
        by1 = max(0, y - th - pad * 2)
        bx2 = x + tw + pad * 2
        by2 = y

        cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(
            img,
            text,
            (bx1 + pad, by2 - pad - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: Optional[float] = None,
        radar_img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw annotations strictly meeting requirements:
        - Label format: "Cone, 1.5m", "Stop Sign, 3.2m", "Barrier, 2.8m"
        - Visual HUD telemetry
        - Optional BEV Radar map overlay in top-right
        """
        out = frame.copy()
        h_frame, w_frame = out.shape[:2]
        robot_contact = (w_frame // 2, h_frame - 10)

        closest_dist = float("inf")
        closest_name = "None"

        for det in detections:
            bbox = [int(v) for v in det["bbox"]]
            dist = det.get("z", det.get("distance", 0.0))
            color = self.get_risk_color(dist)

            if dist < closest_dist:
                closest_dist = dist
                closest_name = det.get("class_name", "object")

            # Ground ray from robot origin to object base
            if self.draw_ground_rays:
                obj_base = ((bbox[0] + bbox[2]) // 2, bbox[3])
                overlay = out.copy()
                cv2.line(overlay, robot_contact, obj_base, color, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

            # Draw aesthetic corner box
            self.draw_corner_box(out, bbox, color, thickness=2)

            # Strict Required Label Format: "Cone, 1.5m" or "Stop Sign, 3.2m"
            label = det.get("formatted_label")
            if not label:
                cname = det.get("class_name", "Object")
                display_name = {
                    "cone": "Cone",
                    "barrier": "Barrier",
                    "stop_sign": "Stop Sign",
                    "stopsign": "Stop Sign"
                }.get(cname.lower().replace(" ", "_"), cname.title())
                label = f"{display_name}, {dist:.1f}m"

            self.draw_text_badge(out, label, (bbox[0], max(20, bbox[1] - 4)), color)

        # Draw Telemetry Header Bar
        cv2.rectangle(out, (0, 0), (w_frame, 38), (20, 24, 28), -1)
        cv2.line(out, (0, 38), (w_frame, 38), (60, 70, 80), 1)

        # Telemetry Texts
        hud_fps = f"FPS: {fps:.1f}" if fps else "EDGE INFERENCE"
        cv2.putText(out, f"ROBOTICS PERCEPTION | {hud_fps}", (14, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

        objs_text = f"Tracked: {len(detections)}"
        cv2.putText(out, objs_text, (w_frame // 2 - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 220), 1, cv2.LINE_AA)

        # Status alert
        if closest_dist < 2.0:
            status_text = f"CRITICAL: {closest_name.upper()} AT {closest_dist:.1f}m"
            alert_color = (0, 0, 240)
        elif closest_dist < 5.0:
            status_text = f"CAUTION: {closest_name.upper()} AT {closest_dist:.1f}m"
            alert_color = (0, 215, 255)
        else:
            status_text = "PATH CLEAR"
            alert_color = (50, 220, 50)

        cv2.putText(out, status_text, (w_frame - 340, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, alert_color, 2, cv2.LINE_AA)

        # Overlay Bird's-Eye View (BEV) radar inset if provided
        if self.draw_bev_inset and radar_img is not None:
            inset_w, inset_h = 220, 220
            resized_radar = cv2.resize(radar_img, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

            rx1 = w_frame - inset_w - 14
            ry1 = 48
            rx2 = rx1 + inset_w
            ry2 = ry1 + inset_h

            # Alpha blend inset into frame
            sub = out[ry1:ry2, rx1:rx2]
            blended = cv2.addWeighted(resized_radar, 0.88, sub, 0.12, 0)
            out[ry1:ry2, rx1:rx2] = blended
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 230, 255), 2)

        return out
