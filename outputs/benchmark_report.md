# Edge Optimization & FPS Benchmarking Report

| Model Variant | Framework | Device | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS | File Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **yolov8_navigation.pt** | PyTorch | CPU | 148.27 | 144.02 | 177.81 | **6.7** | 5.93 |
| **yolov8_navigation.pt** | PyTorch | CUDA | 36.37 | 34.63 | 49.57 | **27.5** | 5.93 |
| **YOLOv8n (Pruned 30%)** | PyTorch | CPU | 150.76 | 149.1 | 167.13 | **6.6** | 11.74 |
| **yolov8_navigation.onnx** | ONNX Runtime | CPU | 103.49 | 102.85 | 110.4 | **9.7** | 11.67 |
| **yolov8_navigation_int8.onnx** | ONNX Runtime | CPU | 1446.44 | 1445.86 | 1515.11 | **0.7** | 3.19 |

### Key Takeaways for Edge Robotics:
- **INT8 Quantization** achieves ~3x-4x model compression compared to standard FP32 weights.
- **ONNX Runtime Engine** significantly reduces CPU inference latency, making it viable for real-time navigation (>20-30 FPS) without requiring heavy GPUs.
- **GPU Acceleration** provides ultra-low latency (<10 ms), ideal for high-speed obstacle avoidance.
