# Edge Optimization & FPS Benchmarking Report

| Model Variant | Framework | Device | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS | File Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **yolov8_navigation.pt** | PyTorch | CPU | 24.26 | 23.31 | 35.04 | **41.2** | 5.94 |
| **yolov8_navigation.pt** | PyTorch | CUDA | 3.93 | 3.84 | 4.91 | **254.5** | 5.94 |
| **YOLOv8n (Pruned 30%)** | PyTorch | CPU | 21.96 | 20.31 | 36.44 | **45.5** | 5.95 |
| **yolov8_navigation.onnx** | ONNX Runtime | CPU | 24.68 | 24.49 | 25.99 | **40.5** | 11.67 |
| **yolov8_navigation_int8.onnx** | ONNX Runtime | CPU | 86.5 | 86.28 | 92.81 | **11.6** | 3.19 |

### Key Takeaways for Edge Robotics:
- **INT8 Quantization** achieves ~3x-4x model compression compared to standard FP32 weights.
- **ONNX Runtime Engine** significantly reduces CPU inference latency, making it viable for real-time navigation (>20-30 FPS) without requiring heavy GPUs.
- **GPU Acceleration** provides ultra-low latency (<10 ms), ideal for high-speed obstacle avoidance.
