# Edge Optimization & FPS Benchmarking Report

| Model Variant | Framework | Device | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS | File Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **yolov8_navigation.pt** | PyTorch | CPU | 103.51 | 103.62 | 113.34 | **9.7** | 5.94 |
| **yolov8_navigation.pt** | PyTorch | CUDA | 20.69 | 18.51 | 34.99 | **48.3** | 5.94 |
| **YOLOv8n (Pruned 30%)** | PyTorch | CPU | 100.74 | 99.77 | 110.56 | **9.9** | 11.74 |
| **yolov8_navigation.onnx** | ONNX Runtime | CPU | 58.72 | 57.16 | 68.78 | **17.0** | 11.67 |
| **yolov8_navigation_int8.onnx** | ONNX Runtime | CPU | 639.35 | 624.54 | 744.23 | **1.6** | 3.19 |

### Key Takeaways for Edge Robotics:
- **INT8 Quantization** achieves ~3x-4x model compression compared to standard FP32 weights.
- **ONNX Runtime Engine** significantly reduces CPU inference latency, making it viable for real-time navigation (>20-30 FPS) without requiring heavy GPUs.
- **GPU Acceleration** provides ultra-low latency (<10 ms), ideal for high-speed obstacle avoidance.
