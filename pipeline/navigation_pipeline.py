"""
End-to-End Robotics Navigation Pipeline.
Integrates Object Detection, Geometric Distance Estimation,
BEV Homography Mapping, Optical Flow Tracking, and Telemetry Annotation.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from models.detector import NavigationDetector
from geometry.distance_estimator import DistanceEstimator
from geometry.homography import BirdEyeViewTransformer
from geometry.optical_flow import ObstacleTracker
from pipeline.annotator import NavigationAnnotator


class RoboticsNavigationPipeline:
    """
    Comprehensive real-time vision perception pipeline for mobile robots.
    Processes camera streams and outputs metric obstacle detections and guidance telemetry.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8_navigation.pt",
        backend: str = "auto",
        camera_config: str = "configs/camera_params.yaml",
        device: Optional[str] = None,
        enable_bev: bool = True,
        enable_tracking: bool = True,
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.45
    ):
        self.detector = NavigationDetector(
            model_path=model_path,
            backend=backend,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device
        )
        self.distance_estimator = DistanceEstimator(camera_config)
        self.enable_bev = enable_bev
        self.enable_tracking = enable_tracking

        if self.enable_bev:
            self.bev = BirdEyeViewTransformer(
                img_width=self.distance_estimator.image_width,
                img_height=self.distance_estimator.image_height
            )
        else:
            self.bev = None

        if self.enable_tracking:
            self.tracker = ObstacleTracker(fps=30.0)
        else:
            self.tracker = None

        self.annotator = NavigationAnnotator(draw_bev_inset=self.enable_bev)
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        estimation_method: str = "hybrid",
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single camera frame.

        Args:
            frame: BGR numpy image (H, W, 3)
            estimation_method: "hybrid", "pinhole", or "ground"
            dt: Optional elapsed time (seconds) since previous frame for tracking / FPS

        Returns:
            annotated_frame: Rendered image with boxes, distance tags, and BEV map
            detections: List of enriched detection dictionaries
            telemetry: Pipeline metadata including FPS and safety status
        """
        start_t = time.perf_counter()

        # 1. Object Detection
        raw_detections = self.detector.detect(frame)

        # 2. Geometric Distance Estimation
        enriched_detections = []
        for det in raw_detections:
            cname = det["class_name"].lower().replace(" ", "_")
            bbox = det["bbox"]

            dist_info = self.distance_estimator.estimate_distance(
                class_name=cname,
                bbox=bbox,
                method=estimation_method
            )

            merged = {**det, **dist_info}
            enriched_detections.append(merged)

        # Elapsed time handling
        curr_time = time.time()
        if dt is not None and isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0:
            effective_dt = float(dt)
        else:
            diff = curr_time - self.prev_time
            effective_dt = diff if diff > 1e-5 else (1.0 / 30.0)
        self.prev_time = curr_time

        # 3. Temporal Optical Flow Tracking & TTC (if enabled)
        if self.enable_tracking and self.tracker is not None:
            self.tracker.dt = effective_dt
            enriched_detections = self.tracker.update(frame, enriched_detections, dt=effective_dt)

        # 4. Bird's-Eye View Top-Down Radar Map (if enabled)
        radar_img = None
        if self.enable_bev and self.bev is not None:
            radar_img = self.bev.render_radar_grid(enriched_detections)

        # Latency & FPS calculation
        proc_time = time.perf_counter() - start_t
        if effective_dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / effective_dt) if self.fps > 0 else (1.0 / effective_dt)

        # 5. Annotation & HUD Overlay
        annotated_frame = self.annotator.annotate(
            frame=frame,
            detections=enriched_detections,
            fps=self.fps,
            radar_img=radar_img
        )

        # Closest obstacle telemetry
        closest_dist = min([d["z"] for d in enriched_detections]) if enriched_detections else float("inf")
        telemetry = {
            "fps": round(self.fps, 1),
            "latency_ms": round(proc_time * 1000.0, 2),
            "num_obstacles": len(enriched_detections),
            "closest_distance_m": round(closest_dist, 2) if closest_dist != float("inf") else None,
            "status": "CRITICAL" if closest_dist < 2.0 else ("CAUTION" if closest_dist < 5.0 else "NOMINAL")
        }

        return annotated_frame, enriched_detections, telemetry
