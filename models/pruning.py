"""
Model Pruning for Edge Robotics Optimization.
Applies L1-norm unstructured and structured channel pruning to convolutional layers
to reduce computational footprint and accelerate inference on edge devices.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def calculate_model_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """Calculate total parameters, zero parameters, and global sparsity percentage."""
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name and isinstance(param, torch.Tensor):
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    sparsity = (zero_params / max(1, total_params)) * 100.0
    return total_params, zero_params, sparsity


def prune_model(
    model_or_path: Any,
    amount: float = 0.30,
    prune_type: str = "l1_unstructured",
    output_path: Optional[str] = "models/yolov8_pruned.pt"
) -> Dict[str, Any]:
    """
    Apply weight pruning to Conv2d layers of the model.

    Args:
        model_or_path: YOLO model instance, PyTorch nn.Module, or path to .pt file
        amount: Fraction of connections/channels to prune (e.g. 0.30 for 30%)
        prune_type: "l1_unstructured" or "ln_structured"
        output_path: Where to save the pruned weights

    Returns:
        Summary metrics dictionary (initial sparsity, post-prune sparsity, total params)
    """
    if isinstance(model_or_path, str):
        if YOLO is None:
            raise ImportError("Ultralytics required to load YOLO weights.")
        yolo_wrapper = YOLO(model_or_path)
        torch_model = yolo_wrapper.model
    elif hasattr(model_or_path, "model") and isinstance(model_or_path.model, nn.Module):
        yolo_wrapper = model_or_path
        torch_model = model_or_path.model
    else:
        yolo_wrapper = None
        torch_model = model_or_path

    if not isinstance(amount, (int, float)) or not (0.0 <= amount < 1.0):
        raise ValueError(f"Pruning amount must be a float between 0.0 and 1.0, got {amount}")
    if prune_type not in ["l1_unstructured", "ln_structured"]:
        raise ValueError(f"Unsupported prune_type '{prune_type}'. Must be 'l1_unstructured' or 'ln_structured'.")

    # Initial stats
    total_pre, zero_pre, spar_pre = calculate_model_sparsity(torch_model)

    pruned_layers_count = 0
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Exclude detection head (model.22) and stem (model.0.conv) to protect bounding box precision
            # Crucially protect model.22.dfl.conv and regression/classification conv heads
            name_parts = name.split(".")
            is_stem = (
                name == "model.0.conv"
                or name.startswith("model.0.")
                or name.startswith("0.")
                or ("0" in name_parts and name_parts.index("0") <= 1)
            )
            is_head = (
                "model.22" in name
                or ".22." in name
                or name.startswith("22.")
                or "22" in name_parts
                or "dfl" in name
            )
            if is_stem or is_head:
                continue

            if prune_type == "l1_unstructured":
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")  # Make pruning permanent
                pruned_layers_count += 1
            elif prune_type == "ln_structured":
                # Prune filters along dimension 0
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                prune.remove(module, "weight")
                pruned_layers_count += 1

    total_post, zero_post, spar_post = calculate_model_sparsity(torch_model)

    # Save pruned model if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Save weights in torch.float16 to maintain compact ~5.94 MB size
        torch_model.half()
        if yolo_wrapper is not None or hasattr(torch_model, "names") or "DetectionModel" in type(torch_model).__name__:
            # Save via PyTorch state dict or torch.save
            torch.save({"model": torch_model}, output_path)
        else:
            torch.save(torch_model.state_dict(), output_path)
        print(f"[Pruning] Saved pruned model to {output_path}")

    summary = {
        "prune_type": prune_type,
        "amount": amount,
        "layers_pruned": pruned_layers_count,
        "total_parameters": total_pre,
        "zero_parameters_before": zero_pre,
        "sparsity_before_pct": round(spar_pre, 2),
        "zero_parameters_after": zero_post,
        "sparsity_after_pct": round(spar_post, 2),
        "output_path": output_path
    }

    print(f"[Pruning] Sparsity changed from {spar_pre:.2f}% -> {spar_post:.2f}% across {pruned_layers_count} Conv2d layers.")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prune YOLO model for edge robotics")
    parser.add_argument("--model", type=str, default="models/yolov8_navigation.pt", help="Path to model")
    parser.add_argument("--amount", type=float, default=0.30, help="Pruning ratio (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="models/yolov8_pruned.pt", help="Output path")
    args = parser.parse_args()

    prune_model(args.model, amount=args.amount, output_path=args.output)
