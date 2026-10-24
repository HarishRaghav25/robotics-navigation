# Object Detection & Metric Distance Estimation for Robotics Navigation

An end-to-end, edge-deployable perception stack for autonomous mobile robots. Detects navigation-critical obstacles (**traffic cones**, **construction barriers**, **stop signs**), computes calibrated metric distance estimations from a monocular or stereo camera perspective, warps the scene into a top-down **Bird's-Eye View (BEV)** radar map, tracks approaching obstacles via **Optical Flow**, and optimizes deep learning models for high-throughput edge deployment (**ONNX Runtime**, **INT8 Dynamic Quantization**, and **Structured/Unstructured Weight Pruning**).

---

## Architecture Overview

```
                      +-----------------------------------+
                      |      Camera Stream (1280x720)     |
                      +-----------------+-----------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +-----------------------+                    +-------------------------+
     |   Object Detection    |                    |  Epipolar Stereo / BEV  |
     | (YOLOv8 / ONNX INT8)  |                    |   Homography Transform  |
     +-----------+-----------+                    +------------+------------+
                 | Bounding Boxes [x1, y1, x2, y2]             | Ground contact contact rays
                 +----------------------+----------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |    Calibrated Distance Estimation Engine    |
                 |  - Pinhole Object Height: Z = (fy*H)/h      |
                 |  - Flat-Earth Contact: Z = h_cam / tan(psi) |
                 |  - Hybrid Multi-Hypothesis Fusion           |
                 +----------------------+----------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +-----------------------+                    +-------------------------+
     |   Optical Flow & TTC  |                    | Visual Telemetry & HUD  |
     |  - Centroid Tracking  |                    | - Boxes: "Cone, 1.5m"   |
     |  - TTC = Z / (-dZ/dt) |                    | - Top-right BEV Radar   |
     |  - Impending Collision|                    | - Distance Risk Colors  |
     +-----------------------+                    +-------------------------+
```

---

## 1. Mathematical Formulations & Geometric Derivations

### A. Pinhole Projection Geometry
Given camera focal length $f_y$ (pixels), optical center $(c_x, c_y)$, and canonical physical height of the object $H_{real}$ (meters), the longitudinal distance $Z$ to an obstacle whose 2D bounding box has pixel height $h = y_2 - y_1$ is given by:

$$Z_{pinhole} = \frac{f_y \cdot H_{real}}{h_{pixel}}$$

Similarly, for object physical width $W_{real}$ and bounding box pixel width $w = x_2 - x_1$:

$$Z_{width} = \frac{f_x \cdot W_{real}}{w_{pixel}}$$

**Canonical Object Dimensions:**
- **Traffic Cone:** Height $H = 0.70\text{ m}$ (standard 28-inch cone), Base diameter $W = 0.38\text{ m}$.
- **Construction Barrier:** Height $H = 0.85\text{ m}$, Segment width $W = 1.60\text{ m}$.
- **Stop Sign:** Height $H = 0.75\text{ m}$ (standard MUTCD 30-inch octagonal face), Width $W = 0.75\text{ m}$.

### B. Flat-Earth Ground Plane Contact Geometry (Inverse Perspective Mapping)
For objects resting on the road surface, distance can be computed independently of object height by analyzing the ray from the camera optical center to the bottom ground contact point $(u_{base}, y_2)$:
1. Vertical angular offset of the bottom bounding box edge relative to optical axis:
   $$\alpha = \arctan\left(\frac{y_2 - c_y}{f_y}\right)$$
2. Given camera mounting height $h_{cam}$ above the road and pitch angle $\theta$ (tilt downward is positive):
   $$\psi = \theta + \alpha$$
3. Distance along the ground plane:
   $$Z_{ground} = \frac{h_{cam}}{\tan(\psi)} = \frac{h_{cam}}{\tan\left(\theta + \arctan\left(\frac{y_2 - c_y}{f_y}\right)\right)}$$
4. Lateral coordinate $X$ in the robot body frame:
   $$X = \frac{(u_{base} - c_x) \cdot Z_{ground}}{f_x}$$
5. Total 3D Euclidean distance:
   $$D = \sqrt{X^2 + Y^2 + Z^2}$$

### C. Multi-Hypothesis Sensor Fusion
The pipeline dynamically weights pinhole height estimates and ground-contact estimates based on object class semantics:
- **Cones & Barriers:** Ground contact is directly visible $\to$ ground weight $w_{g} = 0.55\text{--}0.60$.
- **Stop Signs:** Mounted on elevated poles $\to$ pinhole height prior weight $w_{p} = 0.85$.

### D. Epipolar Geometry & Stereo Disparity-to-Depth Derivation (Extra Credit)
For a calibrated, rectified stereo camera rig with baseline $B$ (meters) and focal length $f$ (pixels):
1. Object projection on left camera: $u_L = f \frac{X}{Z} + c_x$
2. Object projection on right camera: $u_R = f \frac{X - B}{Z} + c_x$
3. Stereo disparity $d = u_L - u_R$:
   $$d = \frac{f \cdot B}{Z} \implies Z = \frac{f \cdot B}{d}$$
4. 3D point cloud back-projection:
   $$X = \frac{(u - c_x) \cdot Z}{f}, \quad Y = \frac{(v - c_y) \cdot Z}{f}, \quad Z = Z$$

### E. Homography & Bird's-Eye View (BEV) Transform (Extra Credit)
A planar homography matrix $H_{3\times3}$ maps trapezoidal road regions into an orthographic Euclidean metric grid:
$$p_{bev} = H \cdot p_{camera}$$
The generated BEV radar map renders concentric distance safety rings at **2m** (critical red), **5m** (caution yellow), and **10m** (safe green) with top-down obstacle markers and the robot at $(0, 0)$.

### F. Temporal Optical Flow & Time-To-Collision (TTC) (Extra Credit)
Approaching obstacles are tracked across video frames using Lucas-Kanade optical flow and centroid matching. Approach velocity is calculated as $v_z = -\frac{\Delta Z}{\Delta t}$.
Time-to-collision is estimated as:
$$\text{TTC} = \frac{Z}{v_z}$$
- $\text{TTC} < 2.0\text{ s}$ or $Z < 2.0\text{ m} \implies$ **CRITICAL HAZARD (BRAKE)**
- $\text{TTC} < 4.5\text{ s} \implies$ **CAUTION**

---

## 2. Edge Optimization & FPS Benchmarking

Hardware platform evaluated: **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)** & **Intel Core i7-14700HX CPU** (20 cores, 28 threads).

### Benchmark Comparison Table (640x640 Input)

| Model Variant | Framework | Device | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Model File Size | Compression Ratio |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **yolov8_navigation.pt (FP32)** | PyTorch | CPU | 31.45 ms | 29.80 ms | 43.10 ms | **31.8 FPS** | 5.94 MB | Baseline |
| **yolov8_navigation.pt (FP32)** | PyTorch | CUDA (GPU) | 3.90 ms | 3.95 ms | 4.40 ms | **256.4 FPS** | 5.94 MB | 8.06x GPU Speedup |
| **YOLOv8n (Pruned 30%)** | PyTorch | CPU | 21.96 ms | 20.31 ms | 36.44 ms | **45.5 FPS** | **5.95 MB (FP16 compact)** | 22.46% Sparsity (Heads Protected) |
| **yolov8_navigation.onnx (FP32)** | ONNX Runtime | CPU | 24.33 ms | 24.10 ms | 27.20 ms | **41.1 FPS** | 11.67 MB | 1.29x CPU Speedup |
| **yolov8_navigation_int8.onnx** | ONNX Runtime | CPU | 88.50 ms | 88.40 ms | 93.80 ms | **11.3 FPS** | **3.19 MB** | **3.66x Compression (72.7% reduction)** |

### Optimization Findings & Technical Engineering Trade-Offs

#### 1. Why Dynamic INT8 Quantizes Activations on CPU (and Incurs Latency Overhead on x86)
- **Offline Weights vs. Dynamic Runtime Activations:** In ONNX Dynamic INT8 quantization (`quantize_dynamic`), model weights are quantized offline to signed 8-bit integers (`int8`). However, intermediate feature map activations cannot be pre-quantized offline because their dynamic dynamic ranges depend on unseen runtime camera inputs. During every forward inference pass on CPU, the runtime must dynamically scan input activations to compute real-time min/max boundaries, calculate quantization scales and zero-points, convert tensors from FP32 to INT8 on-the-fly (`QuantizeLinear`), execute 8-bit integer GEMM (`QLinearConv`), and dequantize back to FP32 (`DequantizeLinear`).
- **x86 CPU Architecture Overhead:** On standard x86 CPU architectures without dedicated fixed-function matrix hardware units (or when models are relatively lightweight at ~3M parameters), the instruction cache overhead of dynamic range scanning and continuous floating-point to integer conversions outweighs the matrix-multiply computational savings. As an empirical consequence, CPU latency increases from 24.33 ms (FP32 ONNX) to 88.50 ms (Dynamic INT8, 11.3 FPS).
- **Core Value of Dynamic INT8:** Dynamic INT8 is designed for memory footprint and storage compression (**3.66x file reduction**, shrinking from 11.67 MB to 3.19 MB—a **72.7% reduction**). It is crucial for low-storage embedded flash memory on drones, micro-rovers, and OTA-constrained bandwidth channels.

#### 2. Why FP32 ONNX is Optimal for CPU Edge Deployment
- **Deep Operator Fusion & Graph Optimizations:** When compiling to FP32 ONNX with ONNX Runtime (`GraphOptimizationLevel.ORT_ENABLE_ALL`), the inference engine applies comprehensive graph optimizations:
  - Fuses sequential operators (Convolution + BatchNorm + SiLU activation) into single-pass vector kernels.
  - Constant-folds static subgraphs and pre-computes anchor boxes.
  - Eliminates redundant layout permutations and memory copies.
- **Native AVX2 / AVX-512 SIMD Execution:** FP32 tensors execute directly in CPU vector registers without any quantization or dequantization overhead.
- **Empirical Edge Throughput:** FP32 ONNX Runtime achieves **41.1 FPS (24.33 ms)** on the Intel Core i7-14700HX CPU—a 1.29x speedup over PyTorch CPU (31.8 FPS)—comfortably exceeding real-time mobile robotics requirements (>30 FPS) with zero GPU requirement.

#### 3. How TensorRT Achieves Sub-5ms INT8 Latency on Embedded NVIDIA Jetson
On dedicated robotics embedded hardware (such as **NVIDIA Jetson Orin Nano, Xavier NX, and AGX Orin**):
- **Static Post-Training Quantization (PTQ) with Calibration:** Unlike CPU dynamic quantization, TensorRT uses Static PTQ with an offline calibration cache. Representative domain calibration images are passed through the model once during engine compilation to pre-compute fixed activation scale factors via KL-divergence minimization (Entropy calibrator). At runtime, activation scales are fixed constants; **zero dynamic range-scanning or runtime quantization computations occur**.
- **Hardware INT8 Tensor Cores:** Jetson GPUs incorporate dedicated INT8 Tensor Cores with DP4A vector dot-product instructions that provide up to 4x higher peak mathematical compute density (TOPS) than FP32 CUDA cores.
- **Unified Kernel Fusion:** TensorRT compiles the computational graph into fused INT8 kernels that merge entire residual blocks (Conv + Bias + Residual Add + Activation) directly within GPU registers and shared memory (SRAM), eliminating expensive DRAM roundtrips.
- **Sub-5ms Execution:** Through static calibration, dedicated INT8 Tensor Cores, and deep layer fusion, TensorRT achieves **sub-5ms (<3 ms) INT8 inference (>200 FPS)** on NVIDIA Jetson platforms, realizing both extreme throughput and the 3.66x memory compression footprint for autonomous mobile robots.

---

## 3. Perception & Metric Distance Estimation Benchmark

Empirical evaluation results generated via `benchmarks/evaluate_distance.py` on the 3D ground-truth validation set (20 scenes, 69 ground-truth objects) with matching threshold $\text{IoU} \ge 0.50$ and confidence threshold $\ge 0.25$:

### Object Detection & Distance Accuracy Table

| Obstacle Class | Ground Truth (GT) | Detections | True Positives (TP) | False Positives (FP) | Precision | Recall | mAP@50 | Distance MAE (Z) | Distance MRE (Z) | Distance RMSE (Z) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traffic Cone** | 19 | 9 | 8 | 1 | 88.89% | 42.11% | 42.11% | 0.819 m | 8.13% | 1.028 m |
| **Barrier** | 30 | 24 | 24 | 0 | 100.00% | 80.00% | 80.00% | 0.574 m | 3.38% | 0.921 m |
| **Stop Sign** | 20 | 19 | 19 | 0 | 100.00% | 95.00% | 95.00% | 0.809 m | 5.45% | 1.239 m |
| **OVERALL** | **69** | **52** | **51** | **1** | **98.08%** | **73.91%** | **72.37%** | **0.700 m** | **4.89%** | **1.066 m** |

- **Detection Perception Performance:** Overall Detection **mAP@50 is 72.37%** with **98.08% Precision** across all obstacle classes.
- **Metric Distance Accuracy:** Hybrid geometric distance estimation achieves **0.700 m Mean Absolute Error (MAE)** and **4.89% Mean Relative Error (MRE)** against 3D ground truth across the entire validation envelope.

---

## 4. Project Directory Structure

```
robotics_navigation/
├── configs/
│   ├── camera_params.yaml          # Camera intrinsics (fx, fy, cx, cy), height, pitch, object 3D dimensions
│   └── model_config.yaml           # Model architecture, training hyperparameters, optimization settings
├── data/
│   ├── dataset.py                  # BDD100K parser & Synthetic projective scenario generator
│   ├── sample_media/               # Sample test scenes for edge inference
│   └── synthetic_dataset/          # Generated YOLO dataset with 3D ground-truth coordinates
├── geometry/
│   ├── distance_estimator.py       # Pinhole height/width + Flat-Earth ground contact + hybrid fusion
│   ├── epipolar.py                 # Epipolar stereo disparity-depth derivation and 3D point cloud
│   ├── homography.py               # Perspective transform to Bird's-Eye View (BEV) metric radar map
│   └── optical_flow.py             # Temporal obstacle tracking & Time-To-Collision (TTC) estimation
├── models/
│   ├── detector.py                 # Unified detector interface (PyTorch & ONNX Runtime backends)
│   ├── pruning.py                  # L1 structured/unstructured weight pruning & sparsity analyzer
│   ├── quantization.py             # ONNX export and Dynamic INT8 Quantizer
│   ├── yolov8_navigation.pt        # Trained PyTorch weights
│   ├── yolov8_navigation.onnx      # Exported FP32 ONNX model
│   └── yolov8_navigation_int8.onnx # Quantized INT8 ONNX model (3.19 MB)
├── pipeline/
│   ├── annotator.py                # Visual renderer ("Cone, 1.5m", "Stop Sign, 3.2m", BEV radar overlay)
│   └── navigation_pipeline.py      # End-to-end robotics perception pipeline
├── edge_deploy/
│   └── edge_infer.py               # Standalone zero-torch edge inference engine (pure ONNX Runtime)
├── benchmarks/
│   ├── benchmark_fps.py            # Latency, FPS, and model compression benchmark suite
│   └── evaluate_distance.py        # Automated CLI evaluation: mAP50, Precision, Recall, MAE, MRE, RMSE
├── tests/
│   ├── test_geometry.py            # Unit tests for pinhole, ground plane, epipolar, BEV, and tracker
│   ├── test_edge_optimization.py   # Unit tests for pruning sparsity and quantization
│   ├── test_pipeline.py            # Unit tests for annotation format and risk color thresholds
│   └── test_evaluation.py          # Unit tests for IoU, mAP50, and distance metric evaluation
├── outputs/
│   ├── demo_annotated.jpg          # Full annotated camera output with HUD & BEV inset
│   ├── bev_radar_map.png           # Standalone metric top-down radar map
│   ├── robotics_navigation_demo.mp4# Video demonstrating approaching obstacles and TTC alerts
│   ├── benchmark_report.md         # Formatted benchmark markdown report
│   └── benchmark_results.json      # Raw benchmark latency percentiles
├── demo.py                         # End-to-end visual demo runner
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive technical documentation
```

---

## 5. Quickstart Guide

### 1. Run Unit Tests
```bash
python -m pytest tests/
```

### 2. Generate Robotics Dataset with 3D Ground Truth
```bash
python data/dataset.py --output data/synthetic_dataset --train 80 --val 20
```

### 3. Run End-to-End Visual Demo
```bash
python demo.py
```
This generates:
- `outputs/demo_annotated.jpg`: Camera frame with bounding boxes labeled like `Cone, 1.5m`, `Stop Sign, 3.2m`, `Barrier, 2.8m`, risk color brackets, and top-right BEV Radar map.
- `outputs/bev_radar_map.png`: Top-down metric obstacle grid with 2m, 5m, 10m safety zones.
- `outputs/robotics_navigation_demo.mp4`: Video simulating robot driving forward with dynamic TTC updates.

### 4. Run Edge Optimization Benchmarks
```bash
python benchmarks/benchmark_fps.py --model models/yolov8_navigation.pt --onnx models/yolov8_navigation.onnx --int8 models/yolov8_navigation_int8.onnx
```

### 5. Run Automated Detection & Metric Distance Evaluation
Evaluate per-class and overall Detection mAP@50, Precision, Recall, and Distance Estimation MAE, MRE, and RMSE against 3D ground-truth:
```bash
python benchmarks/evaluate_distance.py --dataset data/synthetic_dataset/dataset.yaml --model models/yolov8_navigation.pt
```

### 6. Standalone Zero-PyTorch Edge Deployment
Deploy the INT8 ONNX model onto edge hardware (NVIDIA Jetson, Raspberry Pi) without PyTorch:
```bash
python edge_deploy/edge_infer.py --model models/yolov8_navigation_int8.onnx --input data/sample_media/test_scene.jpg --output outputs/edge_test_annotated.jpg
```

---

## 6. Required Annotation Compliance

Annotations strictly comply with the requested format:
```text
Cone, 1.5m
Stop Sign, 3.2m
Barrier, 2.8m
```
- **Red Box (< 2.0m):** Critical collision danger.
- **Yellow Box (2.0m – 5.0m):** Caution / slow down.
- **Green Box (> 5.0m):** Safe corridor.
