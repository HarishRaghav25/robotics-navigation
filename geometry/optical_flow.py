"""
Optical Flow & Motion-based Time-To-Collision (TTC) Estimation.
Tracks moving obstacles (cones, barriers) across video frames and computes
longitudinal approach velocity and Time-To-Collision (TTC) for collision avoidance.
"""

import math
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple


class ObstacleTracker:
    """
    Tracks navigation objects across temporal video frames using Lucas-Kanade
    optical flow and centroid matching, computing approach velocities and TTC.
    """

    def __init__(self, fps: float = 30.0, max_age: int = 5):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.max_age = max_age
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.prev_gray: Optional[np.ndarray] = None

    def compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        bbox: List[int],
        dt: Optional[float] = None
    ) -> Tuple[Tuple[float, float], float, int]:
        """
        Compute sparse Lucas-Kanade optical flow within bounding box.

        Returns:
            flow_vector: (dx, dy) median motion displacement across frames
            ttc_flow: Time-to-collision estimated from optical flow divergence / scale expansion
            num_points: Number of successfully tracked feature points
        """
        effective_dt = dt if (dt is not None and dt > 0) else self.dt
        if prev_gray is None or curr_gray is None or prev_gray.size == 0 or curr_gray.size == 0:
            return (0.0, 0.0), 99.9, 0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = curr_gray.shape
        x1 = max(0, min(w_img - 2, x1))
        x2 = max(x1 + 2, min(w_img, x2))
        y1 = max(0, min(h_img - 2, y1))
        y2 = max(y1 + 2, min(h_img, y2))

        prev_roi = prev_gray[y1:y2, x1:x2]
        if prev_roi.size == 0:
            return (0.0, 0.0), 99.9, 0

        # Extract good Shi-Tomasi feature points inside the obstacle
        corners = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=25,
            qualityLevel=0.03,
            minDistance=4
        )

        if corners is None or len(corners) < 3:
            # Fall back to structured grid points within the bounding box
            # Guard linspace so x2 - x1 < 8 px does not generate inverted coordinates
            # and does not collapse to a single coordinate when x2 - x1 == 8
            gx_start = x1 + 4 if (x2 - x1) > 8 else float(x1)
            gx_end = x2 - 4 if (x2 - x1) > 8 else float(x2)
            gy_start = y1 + 4 if (y2 - y1) > 8 else float(y1)
            gy_end = y2 - 4 if (y2 - y1) > 8 else float(y2)
            gx_end = min(float(w_img - 1), max(gx_start, gx_end))
            gy_end = min(float(h_img - 1), max(gy_start, gy_end))
            gx = np.linspace(gx_start, gx_end, 4)
            gy = np.linspace(gy_start, gy_end, 4)
            grid_pts = np.array([[x, y] for x in gx for y in gy], dtype=np.float32).reshape(-1, 1, 2)
            p0 = grid_pts
        else:
            # Shift corner coordinates from ROI local to global image frame
            p0 = corners + np.array([[[float(x1), float(y1)]]], dtype=np.float32)

        # Calculate Lucas-Kanade Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if p1 is None or st is None:
            return (0.0, 0.0), 99.9, 0

        good_p0 = p0[st == 1]
        good_p1 = p1[st == 1]

        if len(good_p0) < 2:
            return (0.0, 0.0), 99.9, len(good_p0)

        # Median flow displacement vector
        displacements = good_p1 - good_p0
        dx = float(np.median(displacements[:, 0]))
        dy = float(np.median(displacements[:, 1]))

        # Scale expansion rate (looming) for TTC estimation
        # Pairwise distance ratio between tracked points
        ttc_flow = 99.9
        if len(good_p0) >= 4:
            d0_list = []
            d1_list = []
            for i in range(len(good_p0) - 1):
                for j in range(i + 1, min(i + 4, len(good_p0))):
                    dist0 = math.hypot(good_p0[i][0] - good_p0[j][0], good_p0[i][1] - good_p0[j][1])
                    dist1 = math.hypot(good_p1[i][0] - good_p1[j][0], good_p1[i][1] - good_p1[j][1])
                    if dist0 > 4.0:
                        d0_list.append(dist0)
                        d1_list.append(dist1)

            if len(d0_list) >= 3:
                ratios = np.array(d1_list) / np.array(d0_list)
                expansion_rate = float(np.median(ratios)) - 1.0  # (dist_new / dist_old) - 1
                if expansion_rate > 0.005:  # Expanding / approaching
                    ttc_flow = max(0.1, effective_dt / expansion_rate)

        return (round(dx, 2), round(dy, 2)), round(ttc_flow, 1), len(good_p0)

    def update(
        self,
        current_frame: np.ndarray,
        detections: List[Dict[str, Any]],
        dt: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Update obstacle tracks with new frame detections.
        Tracks features using Lucas-Kanade optical flow, computing motion vectors and TTC.
        Supports dynamic dt passed by caller (defaulting to self.dt).
        Maintains track coasting up to max_age frames on intermittent detector drops.
        """
        if current_frame is None:
            return []

        effective_dt = max(1e-4, float(dt) if (dt is not None and dt > 0) else self.dt)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame

        updated_detections = []
        matched_track_ids = set()

        for det in detections:
            if not isinstance(det, dict) or "bbox" not in det or len(det["bbox"]) != 4:
                continue

            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            curr_z = max(0.1, float(det.get("z", det.get("distance", 5.0))))
            curr_x = float(det.get("x", 0.0))
            raw_cls = str(det.get("class_name", "cone")).lower().replace(" ", "_")
            cls = "stop_sign" if raw_cls == "stopsign" else raw_cls

            # Match with closest existing track (guided by optical flow prediction)
            best_id = None
            min_dist = float("inf")

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                track_cls = str(track["class_name"]).lower().replace(" ", "_")
                if track_cls == "stopsign":
                    track_cls = "stop_sign"
                if track_cls != cls:
                    continue

                # Predicted centroid based on optical flow vector or pixel motion
                flow_dx, flow_dy = track.get("flow_vector", (0.0, 0.0))
                if abs(flow_dx) < 0.01 and abs(flow_dy) < 0.01:
                    flow_dx = track.get("v_cx", 0.0) * effective_dt
                    flow_dy = track.get("v_cy", 0.0) * effective_dt

                pred_cx = track["centroid"][0] + flow_dx
                pred_cy = track["centroid"][1] + flow_dy

                dist = math.hypot(cx - pred_cx, cy - pred_cy)
                if dist < 120.0 and dist < min_dist:
                    min_dist = dist
                    best_id = tid

            # Compute Lucas-Kanade optical flow for this obstacle if prev_gray exists
            flow_vec = (0.0, 0.0)
            ttc_flow = 99.9
            num_pts = 0
            if self.prev_gray is not None and best_id is not None:
                prev_bbox = self.tracks[best_id].get("bbox", det["bbox"])
                flow_vec, ttc_flow, num_pts = self.compute_optical_flow(self.prev_gray, gray, prev_bbox, dt=effective_dt)

            if best_id is not None:
                track = self.tracks[best_id]
                prev_z = track["z"]
                # Approach velocity (positive means closing in / approaching)
                v_z = (prev_z - curr_z) / effective_dt
                # Smoothed velocity
                smoothed_vz = 0.7 * v_z + 0.3 * track.get("vz", 0.0)

                # Lateral velocity (vx in body frame)
                prev_x = track.get("x", curr_x)
                v_x = (curr_x - prev_x) / effective_dt
                smoothed_vx = 0.7 * v_x + 0.3 * track.get("vx", 0.0)

                # Pixel velocities
                prev_cx, prev_cy = track["centroid"]
                v_cx = (cx - prev_cx) / effective_dt
                v_cy = (cy - prev_cy) / effective_dt

                # Time to collision (TTC) from depth change
                if smoothed_vz > 0.1:
                    ttc_depth = max(0.1, curr_z / smoothed_vz)
                else:
                    ttc_depth = 99.9

                # Fuse depth-based TTC and optical-flow looming TTC
                if ttc_flow < 50.0 and ttc_depth < 50.0:
                    ttc = 0.60 * ttc_depth + 0.40 * ttc_flow
                elif ttc_depth < 50.0:
                    ttc = ttc_depth
                else:
                    ttc = ttc_flow

                track["centroid"] = (cx, cy)
                track["bbox"] = det["bbox"]
                track["z"] = curr_z
                track["x"] = curr_x
                track["vz"] = smoothed_vz
                track["vx"] = smoothed_vx
                track["v_cx"] = v_cx
                track["v_cy"] = v_cy
                track["ttc"] = ttc
                track["flow_vector"] = flow_vec
                track["age"] += 1
                track["missed_frames"] = 0
                track["last_det"] = det
                matched_track_ids.add(best_id)
                track_id = best_id
            else:
                # Initialize new track
                track_id = self.next_track_id
                self.next_track_id += 1
                smoothed_vz = 0.0
                smoothed_vx = 0.0
                ttc = 99.9
                self.tracks[track_id] = {
                    "class_name": cls,
                    "centroid": (cx, cy),
                    "bbox": det["bbox"],
                    "z": curr_z,
                    "x": curr_x,
                    "vz": smoothed_vz,
                    "vx": smoothed_vx,
                    "v_cx": 0.0,
                    "v_cy": 0.0,
                    "ttc": ttc,
                    "flow_vector": flow_vec,
                    "age": 1,
                    "missed_frames": 0,
                    "last_det": det
                }
                matched_track_ids.add(track_id)

            det_copy = dict(det)
            det_copy["track_id"] = track_id
            det_copy["vz"] = round(smoothed_vz, 2)
            det_copy["vx"] = round(smoothed_vx, 2)
            det_copy["ttc"] = round(ttc, 1)
            det_copy["flow_vector"] = flow_vec
            det_copy["optical_flow_points"] = num_pts

            # Determine safety level
            if ttc < 2.0 or curr_z < 2.0:
                det_copy["hazard_level"] = "CRITICAL"
            elif ttc < 4.5 or curr_z < 5.0:
                det_copy["hazard_level"] = "CAUTION"
            else:
                det_copy["hazard_level"] = "SAFE"

            updated_detections.append(det_copy)

        # Track coasting with max_age = 5: Maintain tracks missed by detector
        # for up to max_age frames, propagating position with last known velocity vector (vx, vz)
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                track = self.tracks[tid]
                track["missed_frames"] = track.get("missed_frames", 0) + 1
                track["age"] = track.get("age", 0) + 1
                if track["missed_frames"] > self.max_age:
                    del self.tracks[tid]
                else:
                    # Propagate position with last known velocity vector (vx, vz)
                    vz = track.get("vz", 0.0)
                    vx = track.get("vx", 0.0)

                    # Longitudinal propagation (approach velocity vz reduces distance)
                    coasted_z = max(0.1, track["z"] - vz * effective_dt)
                    track["z"] = coasted_z

                    # Lateral propagation
                    coasted_x = track.get("x", 0.0) + vx * effective_dt
                    track["x"] = coasted_x

                    # 2D pixel motion propagation
                    flow_dx, flow_dy = track.get("flow_vector", (0.0, 0.0))
                    if abs(flow_dx) < 0.01 and abs(flow_dy) < 0.01:
                        flow_dx = track.get("v_cx", 0.0) * effective_dt
                        flow_dy = track.get("v_cy", 0.0) * effective_dt

                    old_cx, old_cy = track["centroid"]
                    track["centroid"] = (old_cx + flow_dx, old_cy + flow_dy)

                    bx1, by1, bx2, by2 = track["bbox"]
                    track["bbox"] = [bx1 + flow_dx, by1 + flow_dy, bx2 + flow_dx, by2 + flow_dy]

                    # Propagate TTC: maintain continuous TTC without resetting to 99.9s
                    if vz > 0.1:
                        track["ttc"] = max(0.1, coasted_z / vz)
                    else:
                        track["ttc"] = 99.9

                    det_copy = dict(track.get("last_det", {}))
                    det_copy.update({
                        "track_id": tid,
                        "class_name": track["class_name"],
                        "bbox": [round(float(c), 1) for c in track["bbox"]],
                        "z": round(float(coasted_z), 2),
                        "x": round(float(coasted_x), 2),
                        "distance": round(float(coasted_z), 2),
                        "vz": round(float(vz), 2),
                        "vx": round(float(vx), 2),
                        "ttc": round(float(track["ttc"]), 1),
                        "flow_vector": (round(float(flow_dx), 2), round(float(flow_dy), 2)),
                        "optical_flow_points": 0,
                        "coasted": True,
                        "missed_frames": track["missed_frames"]
                    })

                    disp_name = {
                        "cone": "Cone",
                        "barrier": "Barrier",
                        "stop_sign": "Stop Sign",
                        "stopsign": "Stop Sign"
                    }.get(track["class_name"], track["class_name"].title())
                    det_copy["formatted_label"] = f"{disp_name}, {coasted_z:.1f}m"

                    if track["ttc"] < 2.0 or coasted_z < 2.0:
                        det_copy["hazard_level"] = "CRITICAL"
                    elif track["ttc"] < 4.5 or coasted_z < 5.0:
                        det_copy["hazard_level"] = "CAUTION"
                    else:
                        det_copy["hazard_level"] = "SAFE"

                    updated_detections.append(det_copy)

        self.prev_gray = gray.copy()
        return updated_detections
