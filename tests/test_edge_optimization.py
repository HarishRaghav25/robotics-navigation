"""
Unit tests for edge optimization: pruning and quantization modules.
"""

import os
import torch
import torch.nn as nn
import pytest
from models.pruning import prune_model, calculate_model_sparsity


class SimpleTestConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_pruning_sparsity():
    net = SimpleTestConvNet()
    total_p, zero_p, spar_init = calculate_model_sparsity(net)
    assert spar_init == 0.0

    # Apply 30% pruning
    res = prune_model(net, amount=0.30, prune_type="l1_unstructured", output_path=None)
    assert res["sparsity_after_pct"] > 25.0
    assert res["zero_parameters_after"] > 0


def test_onnx_quantization_execution(tmp_path):
    from models.quantization import quantize_onnx_int8
    import onnxruntime as ort

    net = SimpleTestConvNet().eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_file = str(tmp_path / "test_model.onnx")
    int8_file = str(tmp_path / "test_model_int8.onnx")

    # Export to ONNX (embedded weights, no external data files)
    torch.onnx.export(
        net,
        dummy_input,
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False
    )
    assert os.path.exists(onnx_file)

    # Apply INT8 dynamic quantization
    summary = quantize_onnx_int8(onnx_file, int8_file)
    assert os.path.exists(int8_file)
    assert summary["size_int8_mb"] >= 0

    # Verify quantized model runs in ONNX Runtime
    session = ort.InferenceSession(int8_file, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: dummy_input.numpy()})
    assert len(out) == 1
    assert out[0].shape == (1, 32, 32, 32)
