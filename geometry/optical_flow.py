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

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id: int = 1
        self.prev_gray: Optional[np.ndarray] = None

    def update(
        self,
        current_frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update obstacle tracks with new frame detections.
        Computes velocity and TTC for each obstacle.
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame

        updated_detections = []
        matched_track_ids = set()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            curr_z = det.get("z", det.get("distance", 5.0))
            cls = det.get("class_name", "cone")

            # Match with closest existing track
            best_id = None
            min_dist = float("inf")

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if track["class_name"] != cls:
                    continue
                # Euclidean distance between 2D centroids
                prev_cx, prev_cy = track["centroid"]
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist < 100.0 and dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is not None:
                track = self.tracks[best_id]
                prev_z = track["z"]
                # Approach velocity (positive means closing in / approaching)
                v_z = (prev_z - curr_z) / self.dt
                # Smoothed velocity
                smoothed_vz = 0.7 * v_z + 0.3 * track.get("vz", 0.0)

                # Time to collision (TTC) in seconds
                if smoothed_vz > 0.1:  # Approaching
                    ttc = max(0.1, curr_z / smoothed_vz)
                else:
                    ttc = 99.9  # Stationary or receding

                track["centroid"] = (cx, cy)
                track["z"] = curr_z
                track["vz"] = smoothed_vz
                track["ttc"] = ttc
                track["age"] += 1
                matched_track_ids.add(best_id)
                track_id = best_id
            else:
                # Initialize new track
                track_id = self.next_track_id
                self.next_track_id += 1
                smoothed_vz = 0.0
                ttc = 99.9
                self.tracks[track_id] = {
                    "class_name": cls,
                    "centroid": (cx, cy),
                    "z": curr_z,
                    "vz": smoothed_vz,
                    "ttc": ttc,
                    "age": 1
                }
                matched_track_ids.add(track_id)

            det_copy = dict(det)
            det_copy["track_id"] = track_id
            det_copy["vz"] = round(smoothed_vz, 2)
            det_copy["ttc"] = round(ttc, 1)

            # Determine safety level
            if ttc < 2.0 or curr_z < 1.5:
                det_copy["hazard_level"] = "CRITICAL"
            elif ttc < 4.5 or curr_z < 4.0:
                det_copy["hazard_level"] = "CAUTION"
            else:
                det_copy["hazard_level"] = "SAFE"

            updated_detections.append(det_copy)

        # Remove stale tracks
        all_ids = list(self.tracks.keys())
        for tid in all_ids:
            if tid not in matched_track_ids:
                del self.tracks[tid]

        self.prev_gray = gray
        return updated_detections
