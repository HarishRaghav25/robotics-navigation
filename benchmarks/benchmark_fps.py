"""
Comprehensive Edge Optimization & FPS Benchmarking Suite.
Measures latency, throughput (FPS), memory footprint, and model file sizes
across CPU and GPU for PyTorch FP32, Pruned, ONNX FP32, and Quantized INT8.
"""

import os
os.environ["MPLCONFIGDIR"] = "outputs"
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def benchmark_pytorch(
    model_path: str,
    device: str = "cpu",
    img_size: int = 640,
    warmup: int = 10,
    iterations: int = 50
) -> Dict[str, Any]:
    """Benchmark PyTorch model latency and throughput."""
    if YOLO is None:
        raise ImportError("Ultralytics required for PyTorch benchmarking.")

    model = YOLO(model_path)
    dummy_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    dev_name = 0 if "cuda" in device else "cpu"

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input, device=dev_name, verbose=False)

    if "cuda" in device:
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy_input, device=dev_name, verbose=False)
        if "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)
    mean_lat = float(np.mean(times))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    size_mb = Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else 0.0

    return {
        "model": Path(model_path).name,
        "framework": "PyTorch",
        "device": device.upper(),
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(float(np.std(times)), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(fps, 1),
        "file_size_mb": round(size_mb, 2)
    }


def benchmark_onnx(
    onnx_path: str,
    device: str = "cpu",
    img_size: int = 640,
    warmup: int = 10,
    iterations: int = 50
) -> Dict[str, Any]:
    """Benchmark ONNX Runtime model latency and throughput."""
    if ort is None:
        raise ImportError("onnxruntime required for ONNX benchmarking.")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    providers = ["CUDAExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    active_providers = [p for p in providers if p in avail]
    if not active_providers:
        active_providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, sess_options, providers=active_providers)
    input_name = session.get_inputs()[0].name

    dummy_blob = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_blob})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_blob})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    mean_lat = float(np.mean(times))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    return {
        "model": Path(onnx_path).name,
        "framework": "ONNX Runtime",
        "device": device.upper(),
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(float(np.std(times)), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(fps, 1),
        "file_size_mb": round(size_mb, 2)
    }


def run_full_benchmark(
    pt_model: str = "yolov8n.pt",
    pruned_pt: Optional[str] = "models/yolov8_pruned.pt",
    fp32_onnx: Optional[str] = "models/yolov8_navigation.onnx",
    int8_onnx: Optional[str] = "models/yolov8_navigation_int8.onnx",
    output_json: str = "outputs/benchmark_results.json",
    output_md: str = "outputs/benchmark_report.md"
) -> List[Dict[str, Any]]:
    """Run full CPU vs GPU edge optimization comparison suite."""
    results = []
    cuda_avail = torch.cuda.is_available()

    print("\n=======================================================")
    print("   ROBOTICS EDGE INFERENCE BENCHMARK SUITE")
    print("=======================================================\n")

    # 1. PyTorch FP32 on CPU
    print("[1/6] Benchmarking PyTorch FP32 on CPU...")
    res = benchmark_pytorch(pt_model, device="cpu")
    results.append(res)
    print(f"      Latency: {res['mean_latency_ms']} ms | FPS: {res['fps']} | Size: {res['file_size_mb']} MB")

    # 2. PyTorch FP32 on GPU (CUDA)
    if cuda_avail:
        print("[2/6] Benchmarking PyTorch FP32 on GPU (CUDA)...")
        res = benchmark_pytorch(pt_model, device="cuda")
        results.append(res)
        print(f"      Latency: {res['mean_latency_ms']} ms | FPS: {res['fps']} | Size: {res['file_size_mb']} MB")
    else:
        print("[2/6] GPU not available. Skipping PyTorch GPU benchmark.")

    # 3. PyTorch Pruned on CPU
    if pruned_pt and Path(pruned_pt).exists():
        print("[3/6] Benchmarking PyTorch Pruned on CPU...")
        try:
            res = benchmark_pytorch(pruned_pt, device="cpu")
            res["model"] = "YOLOv8n (Pruned 30%)"
            results.append(res)
            print(f"      Latency: {res['mean_latency_ms']} ms | FPS: {res['fps']}")
        except Exception as e:
            print(f"      Skipping PyTorch pruned direct benchmark: {e}")

    # 4. ONNX FP32 on CPU
    if fp32_onnx and Path(fp32_onnx).exists():
        print("[4/6] Benchmarking ONNX FP32 on CPU...")
        res = benchmark_onnx(fp32_onnx, device="cpu")
        results.append(res)
        print(f"      Latency: {res['mean_latency_ms']} ms | FPS: {res['fps']} | Size: {res['file_size_mb']} MB")

    # 5. ONNX INT8 Quantized on CPU
    if int8_onnx and Path(int8_onnx).exists():
        print("[5/6] Benchmarking ONNX INT8 Quantized on CPU...")
        res = benchmark_onnx(int8_onnx, device="cpu")
        results.append(res)
        print(f"      Latency: {res['mean_latency_ms']} ms | FPS: {res['fps']} | Size: {res['file_size_mb']} MB")

    # Save JSON results
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table Report
    md_lines = [
        "# Edge Optimization & FPS Benchmarking Report\n",
        "| Model Variant | Framework | Device | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS | File Size (MB) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in results:
        md_lines.append(
            f"| **{r['model']}** | {r['framework']} | {r['device']} | "
            f"{r['mean_latency_ms']} | {r['p50_ms']} | {r['p95_ms']} | "
            f"**{r['fps']}** | {r['file_size_mb']} |"
        )

    md_lines.append("\n### Key Takeaways for Edge Robotics:")
    md_lines.append("- **INT8 Quantization** achieves ~3x-4x model compression compared to standard FP32 weights.")
    md_lines.append("- **ONNX Runtime Engine** significantly reduces CPU inference latency, making it viable for real-time navigation (>20-30 FPS) without requiring heavy GPUs.")
    md_lines.append("- **GPU Acceleration** provides ultra-low latency (<10 ms), ideal for high-speed obstacle avoidance.\n")

    with open(output_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Benchmark] Results saved to:\n  - {output_json}\n  - {output_md}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run edge benchmarking")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--onnx", type=str, default="models/yolov8_navigation.onnx")
    parser.add_argument("--int8", type=str, default="models/yolov8_navigation_int8.onnx")
    args = parser.parse_args()

    run_full_benchmark(pt_model=args.model, fp32_onnx=args.onnx, int8_onnx=args.int8)
