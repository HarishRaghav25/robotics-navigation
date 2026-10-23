"""
Model Quantization & ONNX Export for Edge Robotics Deployment.
Exports PyTorch weights to ONNX and applies Dynamic INT8 Quantization
to minimize model size, latency, and memory footprint on edge devices (Jetson, Raspberry Pi).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def export_to_onnx(
    model_path: str = "models/yolov8_navigation.pt",
    output_onnx: Optional[str] = "models/yolov8_navigation.onnx",
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False
) -> str:
    """Export YOLO PyTorch weights to ONNX format."""
    if YOLO is None:
        raise ImportError("Ultralytics required for ONNX export.")

    print(f"[Export] Exporting {model_path} to ONNX (opset {opset}, dynamic={dynamic})...")
    yolo = YOLO(model_path)
    exported_file = yolo.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=False
    )

    if output_onnx and Path(exported_file).resolve() != Path(output_onnx).resolve():
        Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(exported_file):
            import shutil
            shutil.copy2(exported_file, output_onnx)
            exported_file = output_onnx

    print(f"[Export] Successfully generated ONNX model: {exported_file}")
    return exported_file


def quantize_onnx_int8(
    input_onnx_path: str,
    output_quant_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quantize ONNX model to INT8 using ONNX Runtime dynamic quantization.
    Achieves ~3-4x model compression with minimal accuracy drop.
    """
    in_path = Path(input_onnx_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")

    if output_quant_path is None:
        output_quant_path = str(in_path.parent / f"{in_path.stem}_int8.onnx")

    out_path = Path(output_quant_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Quantization] Applying Dynamic INT8 Quantization: {input_onnx_path} -> {output_quant_path}")

    # Patch onnx external data handling for Windows path canonicalization compatibility
    try:
        import copy
        import tempfile
        import onnxruntime.quantization.onnx_quantizer as oq_mod
        import onnxruntime.quantization.quant_utils as qu

        def _patched_save_and_reload(model):
            with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
                model_copy = copy.deepcopy(model)
                model_path = Path(quant_tmp_dir) / "model.onnx"
                onnx.save_model(model_copy, str(model_path), save_as_external_data=False)
                return qu.load_model_with_shape_infer(model_path)

        oq_mod.save_and_reload_model_with_shape_infer = _patched_save_and_reload
        qu.save_and_reload_model_with_shape_infer = _patched_save_and_reload
    except Exception as e:
        print(f"[Quantization] Note: quantizer patch skipped ({e})")

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        use_external_data_format=False,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"]
    )

    size_fp32_mb = in_path.stat().st_size / (1024 * 1024)
    size_int8_mb = out_path.stat().st_size / (1024 * 1024)
    reduction_pct = ((size_fp32_mb - size_int8_mb) / size_fp32_mb) * 100.0

    print(f"[Quantization] Completed: {size_fp32_mb:.2f} MB -> {size_int8_mb:.2f} MB ({reduction_pct:.1f}% reduction)")

    return {
        "original_onnx": str(in_path),
        "quantized_onnx": str(out_path),
        "size_fp32_mb": round(size_fp32_mb, 2),
        "size_int8_mb": round(size_int8_mb, 2),
        "reduction_pct": round(reduction_pct, 1),
        "compression_ratio": round(size_fp32_mb / max(0.01, size_int8_mb), 2)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export and quantize YOLO for edge robotics")
    parser.add_argument("--model", type=str, default="models/yolov8_navigation.pt", help="PyTorch model path")
    parser.add_argument("--onnx_out", type=str, default="models/yolov8_navigation.onnx", help="ONNX output")
    parser.add_argument("--int8_out", type=str, default="models/yolov8_navigation_int8.onnx", help="INT8 output")
    args = parser.parse_args()

    onnx_file = export_to_onnx(args.model, args.onnx_out)
    quantize_onnx_int8(onnx_file, args.int8_out)
