"""
Object Detector for Robotics Navigation.
Provides unified inference interface supporting both PyTorch (Ultralytics)
and ONNX Runtime backends for edge deployment.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

CLASS_NAMES = ["cone", "barrier", "stop_sign"]


class NavigationDetector:
    """
    Modular detector supporting transfer-learning trained PyTorch weights
    and optimized ONNX models for edge inference.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8_navigation.pt",
        backend: str = "auto",  # 'pytorch', 'onnx', or 'auto'
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.45,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_names = CLASS_NAMES

        # Device selection
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Backend determination
        if backend == "auto":
            if model_path.endswith(".onnx"):
                self.backend = "onnx"
            else:
                self.backend = "pytorch"
        else:
            self.backend = backend

        self.model = None
        self.ort_session = None
        self._init_backend()

    def _init_backend(self):
        """Initialize the selected backend model."""
        if self.backend == "pytorch":
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics is required for PyTorch backend.")
            print(f"[Detector] Loading PyTorch model: {self.model_path} on {self.device}")
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"[Detector] Failed to load {self.model_path} ({e}), falling back to yolov8n.yaml")
                self.model = YOLO("yolov8n.yaml")
            # Map classes if known
            if hasattr(self.model, "names") and self.model.names:
                if len(self.model.names) == len(CLASS_NAMES):
                    self.class_names = list(self.model.names.values())
        elif self.backend == "onnx":
            if not ORT_AVAILABLE:
                raise ImportError("onnxruntime is required for ONNX backend.")
            print(f"[Detector] Loading ONNX session: {self.model_path}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in self.device else ["CPUExecutionProvider"]
            avail = ort.get_available_providers()
            active_providers = [p for p in providers if p in avail]
            self.ort_session = ort.InferenceSession(self.model_path, providers=active_providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.input_shape = self.ort_session.get_inputs()[0].shape

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 16,
        save_dir: str = "runs/train"
    ) -> str:
        """Fine-tune pretrained YOLO model on robotics dataset via transfer learning."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics required for training.")

        print(f"[Detector] Starting Transfer Learning: {epochs} epochs on {dataset_yaml}")
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if "cuda" in self.device else "cpu",
            project=save_dir,
            name="robotics_nav",
            workers=0,
            exist_ok=True,
            verbose=True
        )

        best_weights = str(Path(save_dir) / "robotics_nav" / "weights" / "best.pt")
        if os.path.exists(best_weights):
            self.model_path = best_weights
            self.model = YOLO(best_weights)
            print(f"[Detector] Transfer learning complete. Best weights: {best_weights}")
            return best_weights
        return self.model_path

    def _preprocess_onnx(self, img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize image for ONNX model inference."""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC BGR to CHW RGB normalized [0, 1]
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_x, pad_y)

    def _postprocess_onnx(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pads: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Parse raw ONNX model output (1, 4+num_classes, N) to bounding boxes with NMS."""
        # Output is typically shape (1, 4 + num_classes, num_anchors)
        out = output[0]  # (channels, anchors)
        if out.shape[0] < out.shape[1]:
            out = out.T  # (anchors, channels)

        boxes = []
        confidences = []
        class_ids = []
        pad_x, pad_y = pads
        orig_h, orig_w = orig_shape

        for row in out:
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf >= self.conf_thresh:
                cx, cy, bw, bh = row[0:4]
                # Map back from letterbox to original image coordinates
                x1 = int(((cx - bw / 2) - pad_x) / scale)
                y1 = int(((cy - bh / 2) - pad_y) / scale)
                x2 = int(((cx + bw / 2) - pad_x) / scale)
                y2 = int(((cy + bh / 2) - pad_y) / scale)

                # Clamp
                x1 = max(0, min(orig_w - 1, x1))
                y1 = max(0, min(orig_h - 1, y1))
                x2 = max(0, min(orig_w - 1, x2))
                y2 = max(0, min(orig_h - 1, y2))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        detections = []
        if len(indices) > 0:
            idxs = indices.flatten() if hasattr(indices, "flatten") else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
            for idx in idxs:
                bx, by, bw, bh = boxes[idx]
                cid = class_ids[idx]
                cname = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(confidences[idx], 3),
                    "bbox": [bx, by, bx + bw, by + bh]
                })

        return detections

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on an image.
        Returns list of detections:
        [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
        """
        h_orig, w_orig = image.shape[:2]

        if self.backend == "pytorch":
            results = self.model(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=0 if "cuda" in self.device else "cpu",
                verbose=False
            )[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": xyxy
                })
            return detections

        elif self.backend == "onnx":
            blob, scale, pads = self._preprocess_onnx(image)
            ort_outs = self.ort_session.run(None, {self.input_name: blob})
            return self._postprocess_onnx(ort_outs[0], (h_orig, w_orig), scale, pads)

        return []
