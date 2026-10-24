"""
Standalone Lightweight Edge Inference Engine for Robotics Navigation.
Zero PyTorch dependency — runs purely on ONNX Runtime, OpenCV, and NumPy.
Designed for embedded robotics controllers: NVIDIA Jetson Nano/Orin, Raspberry Pi 5, Intel NUC.
"""

import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort
import yaml

CLASS_NAMES = ["cone", "barrier", "stop_sign"]

# Canonical object heights and widths (meters)
OBJECT_HEIGHTS = {
    "cone": 0.70,
    "barrier": 0.85,
    "stop_sign": 0.75
}

OBJECT_WIDTHS = {
    "cone": 0.38,
    "barrier": 1.60,
    "stop_sign": 0.75
}


class _DefaultFloat(float):
    pass


_DEFAULT_PITCH = _DefaultFloat(0.0)
_DEFAULT_FX = _DefaultFloat(960.0)
_DEFAULT_FY = _DefaultFloat(960.0)
_DEFAULT_CX = _DefaultFloat(640.0)
_DEFAULT_CY = _DefaultFloat(360.0)
_DEFAULT_CAM_HEIGHT = _DefaultFloat(1.0)


class EdgeInferenceEngine:
    """
    Ultra-lightweight edge detector & geometric distance estimator.
    """

    def __init__(
        self,
        onnx_model_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        fx: float = _DEFAULT_FX,
        fy: float = _DEFAULT_FY,
        cx: float = _DEFAULT_CX,
        cy: float = _DEFAULT_CY,
        cam_height: float = _DEFAULT_CAM_HEIGHT,
        camera_pitch: float = _DEFAULT_PITCH,
        config_path: Optional[str] = "configs/camera_params.yaml"
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.fx = 960.0
        self.fy = 960.0
        self.cx = 640.0
        self.cy = 360.0
        self.image_width = 1280
        self.image_height = 720
        self.cam_height = 1.0
        self.camera_pitch = 0.0

        self.object_heights = dict(OBJECT_HEIGHTS)
        self.object_widths = dict(OBJECT_WIDTHS)
        self.object_heights["stopsign"] = self.object_heights.get("stop_sign", 0.75)
        self.object_widths["stopsign"] = self.object_widths.get("stop_sign", 0.75)

        # Load camera parameters from configs/camera_params.yaml if available
        if config_path:
            p = Path(config_path)
            if not p.is_absolute() and not p.exists():
                repo_root = Path(__file__).resolve().parent.parent
                alt_p = repo_root / config_path
                if alt_p.exists():
                    p = alt_p
            if p.exists():
                self._load_config(str(p))

        # Explicit parameter overrides if provided by caller
        if fx is not _DEFAULT_FX and fx is not None:
            self.fx = float(fx)
        if fy is not _DEFAULT_FY and fy is not None:
            self.fy = float(fy)
        if cx is not _DEFAULT_CX and cx is not None:
            self.cx = float(cx)
        if cy is not _DEFAULT_CY and cy is not None:
            self.cy = float(cy)
        if cam_height is not _DEFAULT_CAM_HEIGHT and cam_height is not None:
            self.cam_height = float(cam_height)
        if camera_pitch is not _DEFAULT_PITCH and camera_pitch is not None:
            self.camera_pitch = float(camera_pitch)

        # Setup ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        avail = ort.get_available_providers()
        active = [p for p in providers if p in avail]

        self.session = ort.InferenceSession(onnx_model_path, sess_options, providers=active)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _load_config(self, config_path: str):
        """Load camera intrinsics, extrinsics, and object dimensions from YAML config."""
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            if not cfg:
                return
            cam = cfg.get("camera", {})
            if "fx" in cam:
                self.fx = float(cam["fx"])
            if "fy" in cam:
                self.fy = float(cam["fy"])
            if "cx" in cam:
                self.cx = float(cam["cx"])
            if "cy" in cam:
                self.cy = float(cam["cy"])
            if "image_width" in cam:
                self.image_width = int(cam["image_width"])
            if "image_height" in cam:
                self.image_height = int(cam["image_height"])
            if "camera_height" in cam:
                self.cam_height = float(cam["camera_height"])
            elif "cam_height" in cam:
                self.cam_height = float(cam["cam_height"])
            if "camera_pitch" in cam:
                self.camera_pitch = float(cam["camera_pitch"])

            objs = cfg.get("object_dimensions", {})
            for name, specs in objs.items():
                norm = name.lower().replace(" ", "_")
                if "height" in specs:
                    self.object_heights[norm] = float(specs["height"])
                if "width" in specs:
                    self.object_widths[norm] = float(specs["width"])
                if norm == "stop_sign":
                    self.object_heights["stopsign"] = self.object_heights[norm]
                    self.object_widths["stopsign"] = self.object_widths[norm]
        except Exception:
            pass

    def preprocess(self, img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox pre-processing."""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_x, pad_y)

    def estimate_distance(self, cls_name: str, bbox: List[int], img_h: Optional[int] = None, img_w: Optional[int] = None) -> float:
        """Fast geometric distance estimation using multi-cue fusion with truncation handling."""
        if bbox is None or len(bbox) != 4:
            return 50.0

        try:
            x1, y1, x2, y2 = bbox
            if any(isinstance(v, bool) for v in [x1, y1, x2, y2]):
                return 50.0
            coords = [float(v) for v in [x1, y1, x2, y2]]
            if not all(np.isfinite(c) for c in coords):
                return 50.0
        except (TypeError, ValueError, IndexError):
            return 50.0

        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return 50.0

        h_orig = img_h if img_h is not None else self.image_height
        w_orig = img_w if img_w is not None else self.image_width

        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        bottom_clipped = y2 >= (h_orig - 3)
        horiz_clipped = (x1 <= 3) or (x2 >= w_orig - 3)

        norm_name = str(cls_name).lower().replace(" ", "_")
        if norm_name == "stopsign":
            norm_name = "stop_sign"

        real_w = self.object_widths.get(norm_name, OBJECT_WIDTHS.get(norm_name, 0.50))
        z_w = (self.fx * real_w) / bw

        aspect = bh / bw
        if norm_name == "stop_sign":
            real_h = 2.05 if aspect > 1.6 else 0.75
        else:
            real_h = self.object_heights.get(norm_name, OBJECT_HEIGHTS.get(norm_name, 0.70))
        z_h = (self.fy * real_h) / bh

        # Ground FOV limit at bottom of image
        dy_bot = max(1.0, float(h_orig - 1 - self.cy))
        alpha_bot = np.arctan2(dy_bot, self.fy)
        total_bot_pitch = self.camera_pitch + alpha_bot
        z_bottom_fov = self.cam_height / np.tan(total_bot_pitch) if total_bot_pitch > 0.01 else 2.67

        # Ground contact ray: when object bottom is at or above horizon (y2 <= cy),
        # ground contact weight is set to 0.0, avoiding 80m fallback inflation.
        if y2 <= self.cy:
            z_ground = 80.0
            ground_valid = False
        else:
            dy = float(y2 - self.cy)
            total_pitch = self.camera_pitch + np.arctan2(dy, self.fy)
            if total_pitch <= 0.01:
                z_ground = 80.0
                ground_valid = False
            else:
                z_ground = self.cam_height / np.tan(total_pitch)
                ground_valid = True

        if bottom_clipped:
            z = min(z_w if not horiz_clipped else z_bottom_fov, z_bottom_fov)
        elif horiz_clipped:
            if ground_valid and (norm_name != "stop_sign" or aspect > 1.6):
                z = 0.60 * z_h + 0.40 * z_ground
            else:
                z = z_h
        elif norm_name == "stop_sign":
            if aspect > 1.6 and ground_valid:
                z = 0.45 * z_w + 0.35 * z_h + 0.20 * z_ground
            else:
                z = 0.50 * z_w + 0.50 * z_h
        elif norm_name == "barrier":
            if ground_valid:
                z = 0.30 * z_w + 0.35 * z_h + 0.35 * z_ground
            else:
                z = (0.30 * z_w + 0.35 * z_h) / 0.65
        else:
            if ground_valid:
                z = 0.35 * z_w + 0.30 * z_h + 0.35 * z_ground
            else:
                # Set ground contact weight to 0.0, avoiding 80m fallback inflation
                z = (0.35 * z_w + 0.30 * z_h) / 0.65

        return float(np.clip(z, 0.3, 50.0))

    def infer_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process frame and return annotated visualization with detections."""
        orig_h, orig_w = frame.shape[:2]
        blob, scale, (pad_x, pad_y) = self.preprocess(frame)

        # Run ONNX inference
        outs = self.session.run(None, {self.input_name: blob})
        raw = outs[0][0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        boxes, confs, class_ids = [], [], []
        for row in raw:
            scores = row[4:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf >= self.conf_thresh:
                cx, cy, bw, bh = row[0:4]
                x1 = int(((cx - bw / 2) - pad_x) / scale)
                y1 = int(((cy - bh / 2) - pad_y) / scale)
                x2 = int(((cx + bw / 2) - pad_x) / scale)
                y2 = int(((cy + bh / 2) - pad_y) / scale)

                x1 = max(0, min(orig_w - 1, x1))
                y1 = max(0, min(orig_h - 1, y1))
                x2 = max(0, min(orig_w - 1, x2))
                y2 = max(0, min(orig_h - 1, y2))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(conf)
                class_ids.append(cid)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.iou_thresh)
        detections = []
        annotated = frame.copy()

        if len(indices) > 0:
            idxs = indices.flatten() if hasattr(indices, "flatten") else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
            for idx in idxs:
                bx, by, bw, bh = boxes[idx]
                cid = class_ids[idx]
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
                x1, y1, x2, y2 = bx, by, bx + bw, by + bh

                dist = self.estimate_distance(cname, [x1, y1, x2, y2], orig_h, orig_w)
                disp_name = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}.get(cname, cname.title())
                label = f"{disp_name}, {dist:.1f}m"

                # Color: Red (<2m), Yellow (2-5m), Green (>5m)
                color = (0, 0, 220) if dist < 2.0 else ((0, 215, 255) if dist < 5.0 else (0, 220, 0))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + len(label) * 11, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                detections.append({
                    "class_name": cname,
                    "confidence": round(confs[idx], 2),
                    "bbox": [x1, y1, x2, y2],
                    "distance_m": round(dist, 2),
                    "label": label
                })

        return annotated, detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Edge Inference Engine")
    parser.add_argument("--model", type=str, default="models/yolov8_navigation_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--input", type=str, default="data/sample_media/test_scene.jpg", help="Input image path")
    parser.add_argument("--output", type=str, default="outputs/edge_output.jpg", help="Output image path")
    args = parser.parse_args()

    engine = EdgeInferenceEngine(args.model)
    img = cv2.imread(args.input)
    if img is not None:
        t0 = time.perf_counter()
        res_img, dets = engine.infer_frame(img)
        dt = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(args.output, res_img)
        print(f"[EdgeInfer] Processed {len(dets)} objects in {dt:.2f} ms ({1000.0/dt:.1f} FPS)")
        for d in dets:
            print(f"  -> {d['label']} (conf: {d['confidence']})")
    else:
        print(f"Could not load image: {args.input}")
