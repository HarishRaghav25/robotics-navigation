"""
Full Robotics Perception & Navigation Demonstration.
Integrates:
1. Object Detection (Cones, Barriers, Stop Signs)
2. Metric Distance Estimation (Geometry Pinhole + Ground Contact)
3. Strict Required Annotations ("Cone, 1.5m", "Stop Sign, 3.2m")
4. Bird's-Eye View (BEV) Metric Radar Map
5. Optical Flow Tracking & Time-to-Collision (TTC)
6. Video Simulation Generation
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import cv2

from data.dataset import SyntheticRoboticsGenerator
from geometry.distance_estimator import DistanceEstimator
from geometry.homography import BirdEyeViewTransformer
from geometry.optical_flow import ObstacleTracker
from pipeline.annotator import NavigationAnnotator
from pipeline.navigation_pipeline import RoboticsNavigationPipeline


def run_demo():
    print("=================================================================")
    print("   ROBOTICS NAVIGATION: DETECTION + DISTANCE ESTIMATION DEMO")
    print("=================================================================\n")

    os.environ["MPLCONFIGDIR"] = "outputs"
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # 1. Initialize Pipeline Modules
    cam_cfg = "configs/camera_params.yaml"
    pipeline = RoboticsNavigationPipeline(
        model_path="models/yolov8_navigation.pt",
        backend="auto",
        camera_config=cam_cfg,
        enable_bev=True,
        enable_tracking=True
    )

    # 2. Generate a high-fidelity test scene with ground-truth objects
    gen = SyntheticRoboticsGenerator(cam_cfg)
    img, ground_truth_objs = gen.generate_scene(scene_id=42)

    # In our test scene, ensure we have at least one cone, barrier, and stop sign
    # to demonstrate exact requested label format
    forced_scene = gen.draw_road_scene()
    demo_objects = [
        {"class_name": "cone", "X": -0.3, "Z": 5.8, "draw": gen.draw_traffic_cone},
        {"class_name": "cone", "X": 0.3, "Z": 3.6, "draw": gen.draw_traffic_cone},
        {"class_name": "stop_sign", "X": 1.4, "Z": 3.0, "draw": gen.draw_stop_sign},
        {"class_name": "barrier", "X": -1.4, "Z": 2.8, "draw": gen.draw_barrier},
    ]

    # Draw distant objects first (Z descending)
    for obj in sorted(demo_objects, key=lambda o: o["Z"], reverse=True):
        obj["draw"](forced_scene, obj["X"], obj["Z"])

    print("[1/4] Detecting navigation-relevant objects & estimating distance:")
    print("-----------------------------------------------------------------")
    t0 = time.perf_counter()
    annotated_frame, detections, telemetry = pipeline.process_frame(forced_scene)
    measured_dt = time.perf_counter() - t0

    for det in detections:
        cname = det["class_name"]
        conf = det.get("confidence", 0.0)
        label = det.get("formatted_label", "")
        z = det.get("z", 0.0)

        # Match with closest ground truth object
        matching_gt = [o for o in demo_objects if o["class_name"] == cname]
        if matching_gt:
            closest_gt = min(matching_gt, key=lambda o: abs(o["Z"] - z))
            err = abs(z - closest_gt["Z"])
            print(f"  * Detected: {label:<18} | Conf: {conf:.2f} | GT Distance: {closest_gt['Z']:.1f}m | Error: {err:.2f}m")
        else:
            print(f"  * Detected: {label:<18} | Conf: {conf:.2f} | Estimated Z: {z:.1f}m")

    # 3. Render Top-Down BEV Radar Map
    print("\n[2/4] Generating Bird's-Eye View (BEV) metric obstacle grid...")
    radar_img = pipeline.bev.render_radar_grid(detections)
    cv2.imwrite("outputs/bev_radar_map.png", radar_img)
    print("      Saved: outputs/bev_radar_map.png")

    # 4. Render Final Annotated Camera Frame
    print("\n[3/4] Rendering annotated navigation frame with HUD telemetry...")
    cv2.imwrite("outputs/demo_annotated.jpg", annotated_frame)
    print("      Saved: outputs/demo_annotated.jpg")

    # 5. Generate Multi-Frame Video Demonstration with Optical Flow TTC
    print("\n[4/4] Generating simulated video sequence with Optical Flow & TTC...")
    video_path = "outputs/robotics_navigation_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

    # Simulate robot driving forward at 0.8 m/s for 40 frames (2.0 seconds)
    speed = 0.8  # m/s
    fps = 20.0
    dt_sim = 1.0 / fps

    # Reset tracker for clean video sequence
    pipeline.tracker = ObstacleTracker(fps=fps)

    cur_objects = [
        {"class_name": "cone", "X": -0.5, "Z": 5.5},
        {"class_name": "barrier", "X": 1.2, "Z": 7.5},
        {"class_name": "stop_sign", "X": 2.2, "Z": 10.5}
    ]

    prev_time = time.perf_counter()
    for frame_idx in range(40):
        frame_img = gen.draw_road_scene()

        # Update Z positions
        for obj in sorted(cur_objects, key=lambda o: o["Z"], reverse=True):
            z_curr = max(0.8, obj["Z"] - speed * (frame_idx * dt_sim))
            x_curr = obj["X"]
            cname = obj["class_name"]

            if cname == "cone":
                gen.draw_traffic_cone(frame_img, x_curr, z_curr)
            elif cname == "stop_sign":
                gen.draw_stop_sign(frame_img, x_curr, z_curr)
            else:
                gen.draw_barrier(frame_img, x_curr, z_curr)

        # Measure elapsed time and process frame with RoboticsNavigationPipeline
        now = time.perf_counter()
        measured_dt = now - prev_time if frame_idx > 0 else dt_sim
        prev_time = now

        rendered_frame, tracked_dets, telem = pipeline.process_frame(frame_img, dt=measured_dt)
        out_video.write(rendered_frame)

    out_video.release()
    print(f"      Saved: {video_path}")

    print("\n=================================================================")
    print("   DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=================================================================\n")


if __name__ == "__main__":
    run_demo()
